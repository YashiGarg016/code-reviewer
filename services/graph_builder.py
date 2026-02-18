import ast
import networkx as nx
from dataclasses import dataclass

@dataclass
class GraphFeatures:
    num_nodes: int
    num_edges: int
    avg_degree: float
    max_degree: int
    density: float
    num_connected_components: int
    avg_shortest_path: float
    clustering_coefficient: float
    most_connected_node: str
    deep_nesting_count: int

def build_ast_graph(content: str) -> nx.DiGraph | None:
    """
    Convert Python source into a directed graph where:
    - Nodes = functions, classes, or modules
    - Edges = call relationships or containment
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    graph = nx.DiGraph()

    # ── Pass 1: Add all function/class nodes ──────────────────────────────
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            graph.add_node(node.name, type="function", line=node.lineno)
        elif isinstance(node, ast.ClassDef):
            graph.add_node(node.name, type="class", line=node.lineno)

    # Always add a module-level node as the root
    graph.add_node("__module__", type="module", line=0)

    # ── Pass 2: Add edges ─────────────────────────────────────────────────
    for node in ast.walk(tree):
        # Class → method containment edges
        if isinstance(node, ast.ClassDef):
            graph.add_edge("__module__", node.name, relation="contains")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    graph.add_edge(node.name, item.name, relation="method")

        # Function call edges
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            caller = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    # Direct function calls: foo()
                    if isinstance(child.func, ast.Name):
                        callee = child.func.id
                        if callee in graph.nodes and callee != caller:
                            graph.add_edge(caller, callee, relation="calls")
                    # Method calls: self.foo() or obj.method()
                    elif isinstance(child.func, ast.Attribute):
                        callee = child.func.attr
                        if callee in graph.nodes and callee != caller:
                            graph.add_edge(caller, callee, relation="calls")

    return graph


def extract_graph_features(content: str) -> GraphFeatures:
    """Extract numeric graph metrics for use as ML features"""
    graph = build_ast_graph(content)

    if graph is None or graph.number_of_nodes() == 0:
        return GraphFeatures(
            num_nodes=0, num_edges=0, avg_degree=0, max_degree=0,
            density=0, num_connected_components=0, avg_shortest_path=0,
            clustering_coefficient=0, most_connected_node="none",
            deep_nesting_count=0
        )

    undirected = graph.to_undirected()

    # Degree stats
    degrees     = dict(graph.degree())
    avg_degree  = sum(degrees.values()) / max(len(degrees), 1)
    max_degree  = max(degrees.values()) if degrees else 0
    most_connected = max(degrees, key=degrees.get) if degrees else "none"

    # Connectivity
    components = nx.number_connected_components(undirected)

    # Average shortest path (only on largest connected component)
    try:
        largest_cc  = max(nx.connected_components(undirected), key=len)
        subgraph    = undirected.subgraph(largest_cc)
        avg_path    = nx.average_shortest_path_length(subgraph) if len(subgraph) > 1 else 0
    except Exception:
        avg_path = 0

    # Clustering coefficient — how interconnected neighbours are
    try:
        clustering = nx.average_clustering(undirected)
    except Exception:
        clustering = 0

    # Deep nesting — count nodes reachable via 4+ hops from module root
    try:
        path_lengths   = nx.single_source_shortest_path_length(graph, "__module__")
        deep_nesting   = sum(1 for depth in path_lengths.values() if depth >= 4)
    except Exception:
        deep_nesting = 0

    return GraphFeatures(
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        avg_degree=round(avg_degree, 3),
        max_degree=max_degree,
        density=round(nx.density(graph), 4),
        num_connected_components=components,
        avg_shortest_path=round(avg_path, 3),
        clustering_coefficient=round(clustering, 3),
        most_connected_node=most_connected,
        deep_nesting_count=deep_nesting
    )