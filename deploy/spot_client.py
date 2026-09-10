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
    """

    def __init__(
        self,
        hostname,
        control_hz=10.0,
        vel_limits=(0.6, 0.4, 0.8),
        reverse_limit=None,
        client_name='horizon_reduction_spot',
    ):
        self.hostname = hostname
        self.control_hz = control_hz
        self.control_period = 1.0 / control_hz
        self.vel_limits = tuple(vel_limits)
        self.reverse_limit = self.vel_limits[0] if reverse_limit is None else reverse_limit
        self.client_name = client_name

        self.robot = None
        self.command_client = None
        self.state_client = None
        self.graph_nav_client = None
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
        self.mobility_params = self._make_mobility_params()

    def _make_mobility_params(self):
        from bosdyn.api import geometry_pb2
        from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
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
        return RobotCommandBuilder.mobility_params(
            vel_limit=vel_limit, stair_hint=spot_command_pb2.HINT_AUTO
        )

    def acquire(self):
        """Register the e-stop, take the lease, power on, and stand."""
        from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
        from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
        from bosdyn.client.robot_command import blocking_stand

        estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        endpoint = EstopEndpoint(client=estop_client, name=self.client_name, estop_timeout=9.0)
        endpoint.force_simple_setup()
        self._estop_keepalive = EstopKeepAlive(endpoint)

        lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self._lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)

        self.robot.power_on(timeout_sec=20)
        blocking_stand(self.command_client, timeout_sec=10)

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
        SDK error string.
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

    # -- GraphNav -----------------------------------------------------------------------------

    def upload_graph(self, map_path):
        """Upload a downloaded map and its snapshots, then localize against the nearest fiducial."""
        from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2

        from deploy.graphnav_map import load_graph, load_snapshots

        graph = load_graph(map_path)
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
