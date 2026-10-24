"""Record a GraphNav map by walking Spot around, with the seed frame zeroed on an AprilTag.

    python deploy/record_map.py --robot_hostname=$SPOT_IP --fiducial_id=301 --out=spot_data_new

Stand Spot roughly square in front of the tag (within ~45° of its face normal) and run this. It
checks the tag is in view, clears whatever map the robot holds, and starts recording. Drive Spot
through the lab with the tablet while the script waits, then press Enter. It stops recording,
closes loops, runs the anchoring optimizer, and writes `graph`, `waypoint_snapshots/` and
`edge_snapshots/` to `--out` in the layout `open_map` and `SpotClient.upload_graph` read.

The zero point: the saved anchoring is re-expressed so the origin sits at the tag, z is up, and +x
points out of the tag face into the room. `seed_tform_body` on any later run is then measured from
the tag, so re-recording the map keeps the same frame as long as the tag has not moved.

The script never takes the lease and never powers the robot on: the tablet holds the lease and
drives. Clearing the robot's map throws away its localization, so the next benchmark run re-uploads
from disk and needs the tag in view, exactly as after a reboot. A new map has new waypoint ids, so
`tasks.json`, `waypoint_xy.json` and `occupancy.npz` from an older map do not carry over.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app, flags

from deploy.spot_client import SpotClient

FLAGS = flags.FLAGS

flags.DEFINE_string('robot_hostname', None, 'Spot address.')
flags.DEFINE_integer('fiducial_id', None, 'AprilTag number that defines the zero point.')
flags.DEFINE_string('out', None, 'Directory to write the map into. Must not already hold one.')
flags.DEFINE_bool('close_loops', True, 'Run odometry and fiducial loop closure before anchoring.')


def zero_frame(tag_pos, tag_rot, robot_pos):
    """Return (x, y, z, yaw) of the tag-zeroed frame expressed in the old seed frame.

    `tag_rot` is the tag's 3x3 rotation (columns are its axes in the seed frame). Which tag axis is
    the face normal depends on the fiducial frame convention, so it is picked from the geometry
    instead: of the six signed axes that lie near the horizontal, the one pointing most directly at
    `robot_pos`, where the robot stood when recording began. z is taken from the robot so the
    ground plane stays near z = 0.
    """
    to_robot = (robot_pos[0] - tag_pos[0], robot_pos[1] - tag_pos[1])
    norm = math.hypot(*to_robot)
    assert norm > 1e-3, 'recording started on top of the tag; start in front of it'
    best, best_score = None, -2.0
    for col in range(3):
        for sign in (1.0, -1.0):
            ax, ay, az = (sign * tag_rot[row][col] for row in range(3))
            horizontal = math.hypot(ax, ay)
            if abs(az) > 0.5:
                continue
            score = (ax * to_robot[0] + ay * to_robot[1]) / (horizontal * norm)
            if score > best_score:
                best, best_score = (ax, ay), score
    assert best is not None, 'no tag axis is near horizontal; the tag must be on a vertical surface'
    return tag_pos[0], tag_pos[1], robot_pos[2], math.atan2(best[1], best[0])


def find_tag(world_object_client, fiducial_id):
    """Return the tag's world object if the robot currently sees it, else None."""
    from bosdyn.api import world_object_pb2

    objects = world_object_client.list_world_objects(
        object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]
    ).world_objects
    for obj in objects:
        if obj.apriltag_properties.tag_id == fiducial_id:
            return obj
    return None


def stop_recording(recording_client, timeout=15.0):
    """Stop recording, retrying while the robot finishes the waypoint it is creating."""
    from bosdyn.client.recording import NotReadyYetError

    deadline = time.time() + timeout
    while True:
        try:
            recording_client.stop_recording()
            return
        except NotReadyYetError:
            if time.time() > deadline:
                raise
            time.sleep(1.0)


def start_waypoint(graph):
    """The first waypoint recorded, which is where Spot stood in front of the tag."""
    return min(
        graph.waypoints,
        key=lambda w: (w.annotations.creation_time.seconds, w.annotations.creation_time.nanos),
    ).id


def rezero(graph, fiducial_id):
    """Re-express every anchor in `graph.anchoring` in the tag-zeroed frame, in place."""
    from bosdyn.client.math_helpers import Quat, SE3Pose

    tag_ids = {str(fiducial_id), f'fiducial_{fiducial_id}', f'filtered_fiducial_{fiducial_id}'}
    tag = [o for o in graph.anchoring.objects if o.id in tag_ids]
    assert tag, (
        f'tag {fiducial_id} is not in the anchoring. Anchored objects: '
        f'{[o.id for o in graph.anchoring.objects]}'
    )
    seed_tform_tag = SE3Pose.from_proto(tag[0].seed_tform_object)

    first = start_waypoint(graph)
    anchor = next(a for a in graph.anchoring.anchors if a.id == first)
    robot = SE3Pose.from_proto(anchor.seed_tform_waypoint)

    x, y, z, yaw = zero_frame(
        (seed_tform_tag.x, seed_tform_tag.y, seed_tform_tag.z),
        seed_tform_tag.rot.to_matrix().tolist(),
        (robot.x, robot.y, robot.z),
    )
    new_tform_old = SE3Pose(x, y, z, Quat.from_yaw(yaw)).inverse()

    for a in graph.anchoring.anchors:
        a.seed_tform_waypoint.CopyFrom(
            new_tform_old.mult(SE3Pose.from_proto(a.seed_tform_waypoint)).to_proto()
        )
    for o in graph.anchoring.objects:
        o.seed_tform_object.CopyFrom(
            new_tform_old.mult(SE3Pose.from_proto(o.seed_tform_object)).to_proto()
        )
    return new_tform_old


def download(graph_nav_client, graph, out):
    """Write the graph and its snapshots in the layout `load_graph` and `load_snapshots` read."""
    os.makedirs(os.path.join(out, 'waypoint_snapshots'), exist_ok=True)
    os.makedirs(os.path.join(out, 'edge_snapshots'), exist_ok=True)
    for waypoint in graph.waypoints:
        if not waypoint.snapshot_id:
            continue
        snapshot = graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
        with open(os.path.join(out, 'waypoint_snapshots', waypoint.snapshot_id), 'wb') as f:
            f.write(snapshot.SerializeToString())
    for edge in graph.edges:
        if not edge.snapshot_id:
            continue
        snapshot = graph_nav_client.download_edge_snapshot(edge.snapshot_id)
        with open(os.path.join(out, 'edge_snapshots', edge.snapshot_id), 'wb') as f:
            f.write(snapshot.SerializeToString())
    # The graph goes last, so a directory holding a `graph` file always holds its snapshots too.
    with open(os.path.join(out, 'graph'), 'wb') as f:
        f.write(graph.SerializeToString())


def main(_):
    assert FLAGS.robot_hostname is not None, 'pass --robot_hostname'
    assert FLAGS.fiducial_id is not None, 'pass --fiducial_id'
    assert FLAGS.out is not None, 'pass --out'
    assert not os.path.exists(os.path.join(FLAGS.out, 'graph')), (
        f'{FLAGS.out} already holds a map; pick a new directory'
    )

    from bosdyn.api.graph_nav import map_processing_pb2
    from bosdyn.client.map_processing import MapProcessingServiceClient
    from bosdyn.client.recording import GraphNavRecordingServiceClient
    from bosdyn.client.world_object import WorldObjectClient
    from google.protobuf import wrappers_pb2

    client = SpotClient(FLAGS.robot_hostname)
    client.connect()
    robot = client.robot
    graph_nav_client = client.graph_nav_client
    recording_client = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
    map_processing_client = robot.ensure_client(MapProcessingServiceClient.default_service_name)
    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)

    if find_tag(world_object_client, FLAGS.fiducial_id) is None:
        print(f'tag {FLAGS.fiducial_id} is not in view. Stand Spot facing it and run again.')
        return
    print(f'tag {FLAGS.fiducial_id} in view')

    graph_nav_client.clear_graph()
    # `require_fiducials` makes the robot refuse to start unless the tag is in the first snapshot,
    # which is what lets the anchoring optimizer place it.
    recording_client.start_recording(require_fiducials=[FLAGS.fiducial_id])
    print('recording. Drive Spot through the lab with the tablet, then come back past the tag.')
    input('press Enter to stop recording... ')
    stop_recording(recording_client)

    if FLAGS.close_loops:
        print('closing loops...')
        map_processing_client.process_topology(
            map_processing_pb2.ProcessTopologyRequest.Params(
                do_odometry_loop_closure=wrappers_pb2.BoolValue(value=True),
                do_fiducial_loop_closure=wrappers_pb2.BoolValue(value=True),
            ),
            modify_map_on_server=True,
        )

    print('optimizing anchoring...')
    map_processing_client.process_anchoring(
        map_processing_pb2.ProcessAnchoringRequest.Params(),
        modify_anchoring_on_server=True,
        stream_intermediate_results=False,
    )

    graph = graph_nav_client.download_graph()
    rezero(graph, FLAGS.fiducial_id)
    download(graph_nav_client, graph, FLAGS.out)
    # The robot's copy keeps the un-zeroed anchoring and the same waypoint ids, so `upload_graph`
    # would skip the upload and every pose would come back in the old frame. Clearing it forces
    # the next run to upload the zeroed map from disk.
    graph_nav_client.clear_graph()

    print(f'wrote {len(graph.waypoints)} waypoints and {len(graph.edges)} edges to {FLAGS.out}')
    first = next(a for a in graph.anchoring.anchors if a.id == start_waypoint(graph))
    p = first.seed_tform_waypoint.position
    print(f'start waypoint at ({p.x:+.2f}, {p.y:+.2f}) m from the tag')


if __name__ == '__main__':
    app.run(main)
