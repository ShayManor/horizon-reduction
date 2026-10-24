"""Boston Dynamics Spot SDK wrapper for the online maze benchmark.

Every `bosdyn` import lives in this file, so the rest of the repo imports without the SDK.

Credentials are read from the environment and never from a flag: `main.py` writes the whole flag
dict to `flags.json` and uploads the same dict to wandb, so a `--robot_password` flag would be
persisted in both.

Field names were written against bosdyn-client 4.x. Confirm them against your SDK version before
the first run; the spec flags this as the one part read from memory rather than from source.
"""
import math
import os
import time


class SpotClient:
    """Lease, e-stop, velocity commands, robot state, and GraphNav localization.

    Args:
        hostname: Spot's address.
        control_hz: Control rate. Velocity commands are issued with an end time slightly past one
            control period so the robot does not stop between steps.
        vel_limits: (forward, lateral, yaw) caps as (m/s, m/s, rad/s). Actions in [-1, 1] scale to
            these. Measure them in your space; the datasheet numbers are not what the robot walks.
        reverse_limit: Backward cap in m/s. Defaults to the forward cap.
        client_name: SDK client name, also the e-stop endpoint name.
        pose_source: 'graphnav' reads the seed-frame pose from GraphNav localization; 'vision' reads
            `vision_tform_body` (visual-inertial odometry), which needs no map.
    """

    def __init__(
        self,
        hostname,
        control_hz=10.0,
        vel_limits=(0.6, 0.4, 0.8),
        reverse_limit=None,
        client_name='horizon_reduction_spot',
        pose_source='graphnav',
    ):
        self.hostname = hostname
        self.control_hz = control_hz
        self.control_period = 1.0 / control_hz
        self.vel_limits = tuple(vel_limits)
        self.reverse_limit = self.vel_limits[0] if reverse_limit is None else reverse_limit
        self.client_name = client_name
        self.pose_source = pose_source

        self.robot = None
        self.command_client = None
        self.state_client = None
        self.graph_nav_client = None
        self.world_object_client = None
        self.mobility_params = None
        self._lease_keepalive = None
        self._estop_keepalive = None
        self._nav_cmd_id = None
        self.last_error = ''

    # -- connection ---------------------------------------------------------------------------

    def connect(self):
        """Authenticate and build the service clients."""
        import bosdyn.client
        from bosdyn.client.graph_nav import GraphNavClient
        from bosdyn.client.robot_command import RobotCommandClient
        from bosdyn.client.robot_state import RobotStateClient
        from bosdyn.client.world_object import WorldObjectClient

        username = os.environ.get('SPOT_USERNAME') or os.environ.get('BOSDYN_CLIENT_USERNAME')
        password = os.environ.get('SPOT_PASSWORD') or os.environ.get('BOSDYN_CLIENT_PASSWORD')
        assert username and password, (
            'Set SPOT_USERNAME and SPOT_PASSWORD in the environment. Credentials must not be flags: '
            'flags are written to flags.json and uploaded to wandb.'
        )

        sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = sdk.create_robot(self.hostname)
        self.robot.authenticate(username, password)
        self.robot.time_sync.wait_for_sync()

        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.world_object_client = self.robot.ensure_client(WorldObjectClient.default_service_name)
        self.mobility_params = self._make_mobility_params()

    def _make_mobility_params(self):
        from bosdyn.api import geometry_pb2
        from bosdyn.client.robot_command import RobotCommandBuilder

        v_fwd, v_lat, w_max = self.vel_limits
        vel_limit = geometry_pb2.SE2VelocityLimit(
            max_vel=geometry_pb2.SE2Velocity(
                linear=geometry_pb2.Vec2(x=v_fwd, y=v_lat), angular=w_max
            ),
            min_vel=geometry_pb2.SE2Velocity(
                linear=geometry_pb2.Vec2(x=-self.reverse_limit, y=-v_lat), angular=-w_max
            ),
        )
        # `mobility_params()` takes no vel_limit argument in bosdyn 5.x: the cap is a field on the
        # MobilityParams message it returns. `stair_hint` is a bool there too, so the old
        # HINT_AUTO enum went in as a truthy stair hint; the lab is flat, so it is dropped.
        params = RobotCommandBuilder.mobility_params()
        params.vel_limit.CopyFrom(vel_limit)
        return params

    def _wait_for_estop_clear(self, estop_client, timeout=10.0):
        """Block until the e-stop reads NONE.

        `EstopKeepAlive` checks in from a background thread, so the robot stays stopped for a
        moment after the endpoint is registered. Powering on inside that window fails.
        """
        from bosdyn.api import estop_pb2

        deadline = time.time() + timeout
        while time.time() < deadline:
            if estop_client.get_status().stop_level == estop_pb2.ESTOP_LEVEL_NONE:
                return
            time.sleep(0.2)
        raise RuntimeError(f'e-stop did not clear within {timeout:.0f}s of registering the endpoint')

    def acquire(self):
        """Take control, register the e-stop, power on, and stand.

        Starts from whatever state the robot was left in: the tablet or a previous run still
        holding the lease, motors already on, or another client's e-stop configuration in place.
        A benchmark that runs four methods back to back cannot ask a person to tidy up in between.
        """
        from bosdyn.client.estop import EstopClient, MotorsOnError
        from bosdyn.client.lease import LeaseClient, LeaseKeepAlive

        # This order matters and is the one that ran a full 25-episode method successfully:
        # e-stop first, then the lease, then power, then stand. Acquiring the lease before
        # reconfiguring the e-stop leaves the command client on a lease the robot supersedes,
        # which shows up much later as STATUS_COMMAND_OVERRIDDEN on the stand.
        estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        lease_client = self.robot.ensure_client(LeaseClient.default_service_name)

        try:
            self._setup_estop(estop_client)
        except MotorsOnError:
            # Motors are on (the operator walked Spot over with the tablet), and the robot refuses an
            # e-stop config change in that state. Rewriting it would mean powering off, i.e. sitting
            # Spot down at every handover, so the current e-stop (the tablet's) is kept as is.
            print('motors already on: keeping the current e-stop configuration')

        try:
            self._lease_keepalive = LeaseKeepAlive(
                lease_client, must_acquire=True, return_at_exit=True
            )
        except Exception:  # noqa: BLE001 - any acquire failure means someone else holds it
            lease_client.take()
            self._lease_keepalive = LeaseKeepAlive(lease_client, return_at_exit=True)

        # Pressing Enter means "take the robot and go": anything another client left that holds the
        # motors off is removed, and power-on is retried after clearing it again.
        # A client that died less than `max_age` ago still blocks; keep retrying until it ages out.
        for attempt in range(10):
            self._clear_keepalive_policies()
            try:
                self.robot.power_on(timeout_sec=20)
                break
            except Exception as exc:  # noqa: BLE001 - retry whatever refused the power-on
                print(f'power on refused ({type(exc).__name__}), clearing and retrying')
                if attempt == 9:
                    raise
                time.sleep(2.0)
        self._stand()

    def _clear_keepalive_policies(self, max_age=10.0):
        """Remove keepalive policies left by dead clients that hold the motors off.

        A client that went away (the tablet after handing over, a killed run) leaves its "motors off
        on comms loss" policy behind, and the robot then refuses power-on (KeepaliveMotorsOffError).
        Only policies that have stopped checking in for `max_age` seconds are removed: the live one
        belongs to this process's own e-stop, and removing it makes the robot sit down.
        """
        from bosdyn.client.keepalive import KeepaliveClient

        client = self.robot.ensure_client(KeepaliveClient.default_service_name)
        now = self.robot.time_sync.robot_timestamp_from_local_secs(time.time()).seconds
        blocking = []
        for live in client.get_status().status:
            if now - live.last_checkin.seconds > max_age and any(
                a.HasField('controlled_motors_off') or a.HasField('immediate_robot_off')
                for a in live.policy.actions
            ):
                blocking.append(live.policy_id)
                print(f'Removing keepalive policy {live.policy.name or live.policy_id!r}')
        if blocking:
            client.modify_policy(policy_ids_to_remove=blocking)

    def _setup_estop(self, estop_client):
        """Register this process as the e-stop endpoint and wait for the stop to clear."""
        from bosdyn.client.estop import EstopEndpoint, EstopKeepAlive

        endpoint = EstopEndpoint(client=estop_client, name=self.client_name, estop_timeout=9.0)
        endpoint.force_simple_setup()
        self._estop_keepalive = EstopKeepAlive(endpoint)
        self._wait_for_estop_clear(estop_client)

    def clear_behavior_faults(self):
        """Clear any behavior faults. The robot refuses every command while one is outstanding.

        An abandoned command or a failed stand leaves a fault behind, and the next run then dies
        on `BehaviorFaultError` with nothing actually wrong. Returns the number cleared.
        """
        faults = self.state_client.get_robot_state().behavior_fault_state.faults
        for fault in faults:
            self.command_client.clear_behavior_fault(fault.behavior_fault_id)
        return len(faults)

    def _stand(self, attempts=3):
        """Stand, clearing behavior faults first and retrying on a transient refusal.

        A behavior fault left by an earlier abandoned command blocks every command until it is
        cleared. The retry deliberately does not touch the lease: taking it again invalidates the
        one this process's own keepalive holds, which is itself a cause of
        STATUS_COMMAND_OVERRIDDEN rather than a cure for it.
        """
        from bosdyn.client.robot_command import (
            BehaviorFaultError,
            CommandFailedErrorWithFeedback,
            blocking_stand,
        )

        for attempt in range(attempts):
            try:
                self.clear_behavior_faults()
                blocking_stand(self.command_client, timeout_sec=10)
                return
            except (BehaviorFaultError, CommandFailedErrorWithFeedback) as exc:
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f'stand failed on every attempt ({type(exc).__name__}). Another client '
                        'holds the robot, or a fault will not clear: check behavior_fault_state '
                        'and whether the tablet has reconnected.'
                    ) from exc
                self.last_error = f'stand failed ({type(exc).__name__}), retrying {attempt + 1}'
                time.sleep(1.0)

    def release(self):
        """Stop the robot, sit, and hand back the lease and e-stop."""
        try:
            self.stop()
            self.robot.power_off(cut_immediately=False, timeout_sec=20)
        finally:
            if self._lease_keepalive is not None:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None
            if self._estop_keepalive is not None:
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None

    # -- action -------------------------------------------------------------------------------

    def scale_action(self, action):
        """Map an action in [-1, 1]^3 to (vx, vy, wz) in m/s and rad/s.

        Forward and reverse caps differ, so the forward axis is scaled by whichever cap the sign
        selects. The result is what actually gets commanded, and is what the raw per-step log
        records next to the achieved velocity.
        """
        a_x, a_y, a_w = (float(min(max(a, -1.0), 1.0)) for a in action)
        v_fwd, v_lat, w_max = self.vel_limits
        vx = a_x * (v_fwd if a_x >= 0.0 else self.reverse_limit)
        return vx, a_y * v_lat, a_w * w_max

    def unscale_velocity(self, vx, vy, wz):
        """Inverse of `scale_action`: express a measured body velocity as an action in [-1, 1]^3.

        GraphNav-driven seed episodes are recorded with the achieved velocity in the action slot,
        since NavigateTo drives the robot itself and never hands back a command.
        """
        v_fwd, v_lat, w_max = self.vel_limits
        a_x = vx / (v_fwd if vx >= 0.0 else self.reverse_limit)
        return (
            min(max(a_x, -1.0), 1.0),
            min(max(vy / v_lat, -1.0), 1.0),
            min(max(wz / w_max, -1.0), 1.0),
        )

    def send_velocity(self, vx, vy, wz, duration=None):
        """Issue one body-frame velocity command."""
        from bosdyn.client.robot_command import RobotCommandBuilder

        duration = 2.0 * self.control_period if duration is None else duration
        command = RobotCommandBuilder.synchro_velocity_command(
            v_x=vx, v_y=vy, v_rot=wz, params=self.mobility_params
        )
        self.command_client.robot_command(command, end_time_secs=time.time() + duration)
        self._nav_cmd_id = None

    def stop(self):
        """Command zero velocity."""
        from bosdyn.client.robot_command import RobotCommandBuilder

        self.command_client.robot_command(RobotCommandBuilder.stop_command())

    # -- state --------------------------------------------------------------------------------

    def get_state(self):
        """Return the fields the env and the raw per-step log need.

        Keys: x, y, yaw (seed frame), vx, vy, wz (body frame), the seven `seed_tform_body`
        components, the GraphNav waypoint id and localization age, battery charge, and the last
        SDK error string. With `pose_source='vision'`, x, y, yaw are in the vision frame, the seed keys
        are absent, and the waypoint id is blank.
        """
        from bosdyn.client.math_helpers import Quat, SE3Pose

        robot_state = self.state_client.get_robot_state()
        kinematic = robot_state.kinematic_state

        # Body-frame velocity: the SDK reports it in the odom frame, so rotate by the inverse of
        # the odom-to-body rotation. This is the controller's tracking readout, not state.
        odom_tform_body = None
        for name, edge in kinematic.transforms_snapshot.child_to_parent_edge_map.items():
            if name == 'body' and edge.parent_frame_name == 'odom':
                odom_tform_body = SE3Pose.from_proto(edge.parent_tform_child)
        linear = kinematic.velocity_of_body_in_odom.linear
        angular = kinematic.velocity_of_body_in_odom.angular
        if odom_tform_body is None:
            vx, vy = linear.x, linear.y
        else:
            body_vel = odom_tform_body.rot.inverse().transform_point(linear.x, linear.y, linear.z)
            vx, vy = body_vel[0], body_vel[1]
        wz = angular.z

        if self.pose_source == 'vision':
            from bosdyn.client.frame_helpers import get_vision_tform_body

            vision_tform_body = get_vision_tform_body(kinematic.transforms_snapshot)
            battery = robot_state.battery_states[0].charge_percentage.value if robot_state.battery_states else 0.0
            return dict(
                x=vision_tform_body.x,
                y=vision_tform_body.y,
                yaw=vision_tform_body.rot.to_yaw(),
                vx=vx,
                vy=vy,
                wz=wz,
                localization_waypoint='',
                localization_age=0.0,
                battery=battery,
                error=self.last_error,
            )

        localization = self.graph_nav_client.get_localization_state().localization
        pose = localization.seed_tform_body
        quat = Quat(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z)

        battery = 0.0
        if robot_state.battery_states:
            battery = robot_state.battery_states[0].charge_percentage.value

        return dict(
            x=pose.position.x,
            y=pose.position.y,
            yaw=quat.to_yaw(),
            vx=vx,
            vy=vy,
            wz=wz,
            seed_x=pose.position.x,
            seed_y=pose.position.y,
            seed_z=pose.position.z,
            seed_qw=pose.rotation.w,
            seed_qx=pose.rotation.x,
            seed_qy=pose.rotation.y,
            seed_qz=pose.rotation.z,
            localization_waypoint=localization.waypoint_id,
            localization_age=time.time() - localization.timestamp.seconds
            if localization.HasField('timestamp')
            else float('nan'),
            battery=battery,
            error=self.last_error,
        )

    def detect_fiducial(self, tag_id=None):
        """The nearest visible AprilTag as (tag_id, (x, y, yaw)) in the vision frame, or None.

        x, y are the tag's position. yaw is the heading of the tag's normal (its z axis) projected on
        the ground, which suits a wall-mounted tag; a tag lying flat falls back to its x axis. With
        `tag_id`, other tags are ignored. Needs no lease.
        """
        from bosdyn.api import world_object_pb2
        from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_a_tform_b, get_vision_tform_body

        objects = self.world_object_client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]
        ).world_objects
        body = get_vision_tform_body(self.state_client.get_robot_state().kinematic_state.transforms_snapshot)
        best = None
        for obj in objects:
            props = obj.apriltag_properties
            if tag_id is not None and props.tag_id != tag_id:
                continue
            frame = props.frame_name_fiducial_filtered or props.frame_name_fiducial
            pose = get_a_tform_b(obj.transforms_snapshot, VISION_FRAME_NAME, frame)
            if pose is None:
                continue
            nx, ny, _ = pose.rot.transform_point(0.0, 0.0, 1.0)
            if math.hypot(nx, ny) < 0.5:
                nx, ny, _ = pose.rot.transform_point(1.0, 0.0, 0.0)
            dist = math.hypot(pose.x - body.x, pose.y - body.y)
            if best is None or dist < best[0]:
                best = (dist, int(props.tag_id), (float(pose.x), float(pose.y), math.atan2(ny, nx)))
        return None if best is None else best[1:]

    # -- GraphNav -----------------------------------------------------------------------------

    def already_localized_to(self, graph):
        """Whether the robot already holds this graph and knows where it is on it."""
        try:
            current = self.graph_nav_client.download_graph()
        except Exception:  # noqa: BLE001 - any failure here just means "re-upload"
            return False
        if current is None:
            return False
        if {w.id for w in current.waypoints} != {w.id for w in graph.waypoints}:
            return False
        return bool(self.graph_nav_client.get_localization_state().localization.waypoint_id)

    def upload_graph(self, map_path, force=False):
        """Upload a downloaded map and its snapshots, then localize against the nearest fiducial.

        Skipped when the robot already holds this graph and is localized on it. `clear_graph`
        throws the localization away, and recovering it needs another fiducial sighting, which the
        robot cannot give from wherever the previous run happened to park it. Re-running this
        between episodes or between methods would therefore fail with STATUS_NO_MATCHING_FIDUCIAL
        anywhere but in front of the tag. Pass `force=True` to upload regardless.
        """
        from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2

        from deploy.graphnav_map import load_graph, load_snapshots

        graph = load_graph(map_path)
        if not force and self.already_localized_to(graph):
            return graph

        waypoint_snapshots, edge_snapshots = load_snapshots(map_path, graph)

        self.graph_nav_client.clear_graph()
        self.graph_nav_client.upload_graph(graph=graph)
        for snapshot in waypoint_snapshots.values():
            self.graph_nav_client.upload_waypoint_snapshot(snapshot)
        for snapshot in edge_snapshots.values():
            self.graph_nav_client.upload_edge_snapshot(snapshot)

        self.graph_nav_client.set_localization(
            initial_guess_localization=nav_pb2.Localization(),
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST,
        )
        return graph

    def navigate_to(self, waypoint_id, command_duration=5.0):
        """Start or continue a GraphNav NavigateTo. Returns the navigation command id."""
        self._nav_cmd_id = self.graph_nav_client.navigate_to(
            waypoint_id, command_duration, command_id=self._nav_cmd_id
        )
        return self._nav_cmd_id

    def navigation_reached(self):
        """Whether the running NavigateTo has reached its goal."""
        from bosdyn.api.graph_nav import graph_nav_pb2

        if self._nav_cmd_id is None:
            return False
        feedback = self.graph_nav_client.navigation_feedback(self._nav_cmd_id)
        return feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL

    def navigate_blocking(self, waypoint_id, timeout=180.0):
        """Drive to a waypoint under GraphNav and wait. Used for episode resets."""
        self._nav_cmd_id = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.navigate_to(waypoint_id)
            time.sleep(0.5)
            if self.navigation_reached():
                self._nav_cmd_id = None
                return True
        self._nav_cmd_id = None
        self.last_error = f'navigate_to({waypoint_id}) timed out after {timeout:.0f}s'
        return False


def yaw_to_cos_sin(yaw):
    """Heading as the (cos, sin) pair the observation carries, which is continuous across +/-pi."""
    return math.cos(yaw), math.sin(yaw)
