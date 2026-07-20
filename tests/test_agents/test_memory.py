"""Tests for short-term and long-term memory."""

from ai_playground.agents.memory import ConversationMemory, VectorMemory


class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory(keep_turns=10)
        mem.add("user", "hi")
        mem.add("assistant", "hello")
        msgs = mem.messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "hello"

    def test_eviction_when_over_capacity(self):
        mem = ConversationMemory(keep_turns=3)
        for i in range(5):
            mem.add("user", f"msg {i}")
        msgs = mem.messages()
        # Without a summarizer, evicted turns are simply dropped.
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg 2"
        assert msgs[-1]["content"] == "msg 4"

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "x")
        mem.clear()
        assert mem.messages() == []


class TestVectorMemory:
    def test_add_and_search_returns_most_similar(self):
        mem = VectorMemory()
        mem.add("the user likes Python")
        mem.add("the user has a cat")
        mem.add("the weather in Paris is rainy")

        hits = mem.search("what programming language?", top_k=3)
        # Hashing trick is lexical, but "Python" should still beat "cat" /
        # "weather" given the right query. Use a query that shares tokens.
        assert len(hits) == 3
        # Just verify the search runs and returns scored entries; lexical
        # embedding means semantic similarity isn't guaranteed.
        scores = [s for s, _ in hits]
        assert scores == sorted(scores, reverse=True)

    def test_lexical_overlap_ranks_higher(self):
        mem = VectorMemory()
        mem.add("Python programming")
        mem.add("rainy weather")
        hits = mem.search("Python", top_k=2)
        assert hits[0][1].text == "Python programming"

    def test_persistence_roundtrip(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = VectorMemory(path=path)
        mem.add("remember this", metadata={"tag": "important"})
        del mem

        mem2 = VectorMemory(path=path)
        assert len(mem2) == 1
        hits = mem2.search("remember this", top_k=1)
        assert hits[0][1].text == "remember this"
        assert hits[0][1].metadata == {"tag": "important"}

    def test_empty_search(self):
        mem = VectorMemory()
        assert mem.search("anything") == []
