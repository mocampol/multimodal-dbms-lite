
class WaitForGraph:


    def __init__(self):
        self._edges: dict[int, set[int]] = {}

    def add_wait(self, waiter: int, holder: int) -> None:
        self._edges.setdefault(waiter, set()).add(holder)

    def remove_txn(self, txn_id: int) -> None:
        self._edges.pop(txn_id, None)
        for waiters in self._edges.values():
            waiters.discard(txn_id)

    def has_cycle(self) -> bool:
        visited: set[int] = set()
        in_stack: set[int] = set()

        def dfs(node: int) -> bool:
            visited.add(node)
            in_stack.add(node)

            for neighbor in self._edges.get(node, set()):
                if neighbor in in_stack:
                    return True
                if neighbor not in visited and dfs(neighbor):
                    return True

            in_stack.discard(node)
            return False

        return any(dfs(node) for node in list(self._edges) if node not in visited)