"""Tests for the MCTS hypothesis tree."""

from __future__ import annotations

import pytest

from core.models import HypothesisStatus
from orchestrator.hypothesis_tree import HypothesisTree


@pytest.fixture()
def tree() -> HypothesisTree:
    return HypothesisTree(session_id="test-session", max_depth=3)


@pytest.fixture()
def populated_tree(tree: HypothesisTree) -> HypothesisTree:
    """Tree with root + 3 children."""
    root = tree.set_root("Root hypothesis", rationale="Base investigation")
    tree.expand(root.id, [
        {"hypothesis": "Hypothesis A", "rationale": "Molecular mechanism"},
        {"hypothesis": "Hypothesis B", "rationale": "Clinical evidence"},
        {"hypothesis": "Hypothesis C", "rationale": "Pathway analysis"},
    ])
    return tree


class TestTreeInitialization:
    def test_set_root(self, tree: HypothesisTree) -> None:
        root = tree.set_root("Test hypothesis")
        assert root.hypothesis == "Test hypothesis"
        assert root.depth == 0
        assert root.status == HypothesisStatus.UNEXPLORED
        assert tree.root is root
        assert tree.node_count == 1

    def test_set_root_with_rationale(self, tree: HypothesisTree) -> None:
        root = tree.set_root("Test", rationale="Because reasons")
        assert root.rationale == "Because reasons"


class TestExpansion:
    def test_expand_root(self, tree: HypothesisTree) -> None:
        root = tree.set_root("Root")
        children = tree.expand(root.id, [
            {"hypothesis": "Child 1"},
            {"hypothesis": "Child 2"},
        ])
        assert len(children) == 2
        assert children[0].parent_id == root.id
        assert children[0].depth == 1
        assert tree.node_count == 3

    def test_expand_nonexistent_raises(self, tree: HypothesisTree) -> None:
        with pytest.raises(Exception, match="not found"):
            tree.expand("nonexistent", [{"hypothesis": "X"}])

    def test_expand_respects_max_depth(self, tree: HypothesisTree) -> None:
        tree.max_depth = 1
        root = tree.set_root("Root")
        children = tree.expand(root.id, [{"hypothesis": "L1"}])
        assert len(children) == 1

        grandchildren = tree.expand(children[0].id, [{"hypothesis": "L2"}])
        assert len(grandchildren) == 0  # blocked by max_depth

    def test_children_accessor(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)
        assert len(children) == 3


class TestUCB1Selection:
    def test_select_unvisited_first(self, populated_tree: HypothesisTree) -> None:
        """Unvisited nodes have infinite UCB1 score."""
        selected = populated_tree.select()
        assert selected.visit_count == 0
        assert selected.status == HypothesisStatus.EXPLORING

    def test_select_after_visits(self, populated_tree: HypothesisTree) -> None:
        """After visiting all children, selects by UCB1."""
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        # Visit each child with different info gains
        populated_tree.backpropagate(children[0].id, 1.0)
        populated_tree.backpropagate(children[1].id, 3.0)
        populated_tree.backpropagate(children[2].id, 0.5)

        # Select — should prefer less-visited or higher UCB1
        selected = populated_tree.select()
        assert selected.status == HypothesisStatus.EXPLORING

    def test_select_no_root_raises(self, tree: HypothesisTree) -> None:
        with pytest.raises(Exception, match="no root"):
            tree.select()


class TestBackpropagation:
    def test_backpropagate_updates_leaf(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.backpropagate(children[0].id, 2.0)
        assert children[0].visit_count == 1
        assert children[0].total_info_gain == 2.0
        assert children[0].avg_info_gain == 2.0

    def test_backpropagate_propagates_to_root(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.backpropagate(children[0].id, 2.0)
        # Root should also be updated
        assert root.visit_count == 1
        assert root.total_info_gain == 2.0

    def test_multiple_backpropagations(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.backpropagate(children[0].id, 2.0)
        populated_tree.backpropagate(children[1].id, 4.0)

        assert root.visit_count == 2
        assert root.total_info_gain == 6.0
        assert root.avg_info_gain == 3.0

    def test_nonexistent_node_raises(self, tree: HypothesisTree) -> None:
        tree.set_root("Root")
        with pytest.raises(Exception, match="not found"):
            tree.backpropagate("nonexistent", 1.0)


class TestInfoGain:
    def test_basic_info_gain(self) -> None:
        gain = HypothesisTree.compute_info_gain(edges_added=5, avg_evidence_quality=0.8)
        assert gain > 0

    def test_falsification_contributes(self) -> None:
        gain_without = HypothesisTree.compute_info_gain(edges_added=5)
        gain_with = HypothesisTree.compute_info_gain(edges_added=5, edges_falsified=2)
        assert gain_with > gain_without

    def test_contradictions_contribute(self) -> None:
        gain_without = HypothesisTree.compute_info_gain(edges_added=5)
        gain_with = HypothesisTree.compute_info_gain(edges_added=5, contradictions_found=2)
        assert gain_with > gain_without

    def test_zero_edges_zero_gain(self) -> None:
        gain = HypothesisTree.compute_info_gain()
        assert gain == 0.0


class TestPruning:
    def test_prune_single_node(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        count = populated_tree.prune(children[0].id)
        assert count == 1
        assert children[0].status == HypothesisStatus.PRUNED

    def test_prune_subtree(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        # Expand child A
        populated_tree.expand(children[0].id, [
            {"hypothesis": "Sub A1"},
            {"hypothesis": "Sub A2"},
        ])

        count = populated_tree.prune(children[0].id)
        assert count == 3  # child + 2 grandchildren

    def test_auto_prune(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        # Give child 0 enough low-gain visits
        for _ in range(4):
            populated_tree.backpropagate(children[0].id, 0.01)

        pruned = populated_tree.auto_prune(min_visits=3, min_avg_gain=0.1)
        assert pruned > 0
        assert children[0].status == HypothesisStatus.PRUNED

    def test_pruned_nodes_skipped_in_selection(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.prune(children[0].id)
        populated_tree.prune(children[1].id)

        selected = populated_tree.select()
        assert selected.id == children[2].id


class TestConfirmationRefutation:
    def test_confirm(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.confirm(children[0].id, 0.9)
        assert children[0].status == HypothesisStatus.CONFIRMED
        assert children[0].confidence == 0.9

    def test_refute(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.refute(children[0].id, "Counter-evidence found")
        assert children[0].status == HypothesisStatus.REFUTED
        assert children[0].confidence == 0.0


class TestTermination:
    def test_terminates_on_max_iterations(self, populated_tree: HypothesisTree) -> None:
        should_stop, reason = populated_tree.should_terminate(
            max_iterations=5, current_iteration=5,
        )
        assert should_stop
        assert "max_iterations" in reason

    def test_continues_with_unexplored(self, populated_tree: HypothesisTree) -> None:
        should_stop, _ = populated_tree.should_terminate(
            max_iterations=15, current_iteration=1,
        )
        assert not should_stop

    def test_terminates_when_all_pruned(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None

        # Prune everything
        for child_id in root.children:
            populated_tree.prune(child_id)
        populated_tree.prune(root.id)

        should_stop, reason = populated_tree.should_terminate(
            max_iterations=15, current_iteration=1,
        )
        assert should_stop
        assert "pruned" in reason

    def test_keeps_exploring_alternatives(self, populated_tree: HypothesisTree) -> None:
        """Per LAB-Bench: don't stop at first confident answer if alternatives exist."""
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.confirm(children[0].id, 0.9)
        # children[1] and [2] are still UNEXPLORED

        should_stop, reason = populated_tree.should_terminate(
            confidence_threshold=0.7, max_iterations=15, current_iteration=1,
        )
        assert not should_stop
        assert "alternatives" in reason


class TestRanking:
    def test_get_best_hypothesis(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.backpropagate(children[0].id, 1.0)
        populated_tree.backpropagate(children[1].id, 5.0)
        populated_tree.backpropagate(children[2].id, 2.0)

        best = populated_tree.get_best_hypothesis()
        assert best is not None
        assert best.id == children[1].id  # highest avg_info_gain=5.0

    def test_ranking_order(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        populated_tree.backpropagate(children[0].id, 1.0)
        populated_tree.backpropagate(children[1].id, 5.0)
        populated_tree.backpropagate(children[2].id, 3.0)

        ranking = populated_tree.get_ranking()
        gains = [n.avg_info_gain for n in ranking]
        assert gains == sorted(gains, reverse=True)


class TestSerialization:
    def test_to_dict(self, populated_tree: HypothesisTree) -> None:
        d = populated_tree.to_dict()
        assert d["node_count"] == 4  # root + 3 children
        assert d["session_id"] == "test-session"
        assert d["root_id"] is not None

    def test_exploration_path(self, populated_tree: HypothesisTree) -> None:
        root = populated_tree.root
        assert root is not None
        children = populated_tree.get_children(root.id)

        path = populated_tree.get_exploration_path(children[0].id)
        assert len(path) == 2
        assert path[0].id == root.id
        assert path[1].id == children[0].id


class TestEvents:
    def test_events_emitted_on_init(self, tree: HypothesisTree) -> None:
        tree.set_root("Test")
        events = tree.drain_events()
        assert len(events) == 1
        assert events[0].event_type == "hypothesis_tree_initialized"

    def test_drain_clears_events(self, tree: HypothesisTree) -> None:
        tree.set_root("Test")
        tree.drain_events()
        assert len(tree.drain_events()) == 0
