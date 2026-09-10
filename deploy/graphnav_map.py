"""Read a downloaded GraphNav map: waypoint positions in the seed frame and graph distances.

The seed frame is the map frame, so waypoint positions read here are directly comparable with the
`seed_tform_body` that `SpotClient.get_state` reports. No registration step is involved.

`bosdyn` is imported lazily so the rest of the repo imports without the SDK installed.
"""
import heapq
import math
import os


def _map_pb2():
    from bosdyn.api.graph_nav import map_pb2

    return map_pb2


def load_graph(map_path):
    """Parse the `graph` file written by GraphNav's map download."""
    graph = _map_pb2().Graph()
    with open(os.path.join(map_path, 'graph'), 'rb') as f:
        graph.ParseFromString(f.read())
    return graph


def load_snapshots(map_path, graph):
    """Load the waypoint and edge snapshots that `upload_graph` has to send alongside the graph."""
    map_pb2 = _map_pb2()

    waypoint_snapshots = {}
    for waypoint in graph.waypoints:
        if not waypoint.snapshot_id:
            continue
        path = os.path.join(map_path, 'waypoint_snapshots', waypoint.snapshot_id)
        if not os.path.exists(path):
            continue
        snapshot = map_pb2.WaypointSnapshot()
        with open(path, 'rb') as f:
            snapshot.ParseFromString(f.read())
        waypoint_snapshots[snapshot.id] = snapshot

    edge_snapshots = {}
    for edge in graph.edges:
        if not edge.snapshot_id:
            continue
        path = os.path.join(map_path, 'edge_snapshots', edge.snapshot_id)
        if not os.path.exists(path):
            continue
        snapshot = map_pb2.EdgeSnapshot()
        with open(path, 'rb') as f:
            snapshot.ParseFromString(f.read())
        edge_snapshots[snapshot.id] = snapshot

    return waypoint_snapshots, edge_snapshots


def _quat_to_yaw(qw, qx, qy, qz):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _compose(a, b):
    """Compose two (x, y, yaw) planar transforms: a_tform_c = a_tform_b * b_tform_c."""
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, ayaw + byaw)


def _edge_transform(edge):
    p = edge.from_tform_to.position
    q = edge.from_tform_to.rotation
    return (p.x, p.y, _quat_to_yaw(q.w, q.x, q.y, q.z))


def waypoint_poses(graph):
    """Return {waypoint_id: (x, y, yaw)} in the seed frame.

    Anchored maps carry `seed_tform_waypoint` directly. Unanchored maps are walked from the first
    waypoint over `from_tform_to`, which puts the seed frame at that waypoint.
    """
    anchors = {a.id: a for a in graph.anchoring.anchors}
    if len(anchors) == len(graph.waypoints) and len(anchors) > 0:
        poses = {}
        for waypoint in graph.waypoints:
            anchor = anchors[waypoint.id]
            p = anchor.seed_tform_waypoint.position
            q = anchor.seed_tform_waypoint.rotation
            poses[waypoint.id] = (p.x, p.y, _quat_to_yaw(q.w, q.x, q.y, q.z))
        return poses

    adjacency = {}
    for edge in graph.edges:
        tform = _edge_transform(edge)
        adjacency.setdefault(edge.id.from_waypoint, []).append((edge.id.to_waypoint, tform))
        inv_yaw = -tform[2]
        c, s = math.cos(inv_yaw), math.sin(inv_yaw)
        inv = (-(c * tform[0] - s * tform[1]), -(s * tform[0] + c * tform[1]), inv_yaw)
        adjacency.setdefault(edge.id.to_waypoint, []).append((edge.id.from_waypoint, inv))

    root = graph.waypoints[0].id
    poses = {root: (0.0, 0.0, 0.0)}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nxt, tform in adjacency.get(cur, []):
            if nxt in poses:
                continue
            poses[nxt] = _compose(poses[cur], tform)
            queue.append(nxt)
    return poses


def edge_lengths(graph):
    """Return {(from_id, to_id): length} in metres, both directions."""
    lengths = {}
    for edge in graph.edges:
        p = edge.from_tform_to.position
        length = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        lengths[(edge.id.from_waypoint, edge.id.to_waypoint)] = length
        lengths[(edge.id.to_waypoint, edge.id.from_waypoint)] = length
    return lengths


def graph_distance(graph, start_id, goal_id):
    """Shortest path length over the waypoint graph, in metres. `inf` if unreachable.

    Use it to pick task pairs whose graph distance substantially exceeds the straight-line
    distance between the same two waypoints: pairs where it does not are solved by driving at the
    goal and tell you nothing about the value function.
    """
    lengths = edge_lengths(graph)
    adjacency = {}
    for (a, b), length in lengths.items():
        adjacency.setdefault(a, []).append((b, length))

    dist = {start_id: 0.0}
    frontier = [(0.0, start_id)]
    while frontier:
        d, cur = heapq.heappop(frontier)
        if cur == goal_id:
            return d
        if d > dist.get(cur, float('inf')):
            continue
        for nxt, length in adjacency.get(cur, []):
            nd = d + length
            if nd < dist.get(nxt, float('inf')):
                dist[nxt] = nd
                heapq.heappush(frontier, (nd, nxt))
    return float('inf')


class GraphNavMap:
    """Waypoint positions and graph distances for one downloaded map."""

    def __init__(self, map_path):
        self.map_path = map_path
        self.graph = load_graph(map_path)
        self.poses = waypoint_poses(self.graph)
        self.short_codes = {}
        for waypoint in self.graph.waypoints:
            self.short_codes.setdefault(waypoint.id[:2], []).append(waypoint.id)
            if waypoint.annotations.name:
                self.short_codes.setdefault(waypoint.annotations.name, []).append(waypoint.id)

    def resolve(self, name):
        """Map a waypoint id, two-letter short code, or annotation name to a full waypoint id."""
        if name in self.poses:
            return name
        candidates = self.short_codes.get(name, [])
        assert len(candidates) == 1, f'waypoint {name!r} resolves to {len(candidates)} waypoints'
        return candidates[0]

    def xy(self, name):
        """Seed-frame (x, y) of a waypoint."""
        x, y, _ = self.poses[self.resolve(name)]
        return (x, y)

    def distance(self, start, goal):
        """Graph distance in metres between two waypoints."""
        return graph_distance(self.graph, self.resolve(start), self.resolve(goal))
