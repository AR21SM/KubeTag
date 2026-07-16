from kubetag.text_processing import encode_head_tail, prepare_text


def test_training_and_runtime_text_contract() -> None:
    labels = ["kind/bug", "sig/node"]
    assert (
        prepare_text(
            "Kubelet failure",
            "See https://example.test/details\n/sig node",
            labels,
        )
        == "Title: Kubelet failure\nBody: See <URL>"
    )


def test_removes_quoted_list_and_backticked_commands() -> None:
    labels = [
        "kind/bug",
        "kind/cleanup",
        "kind/feature",
        "sig/node",
        "area/test",
    ]
    body = """> /kind feature
Keep this `/kind cleanup` text.
- /sig node
1. /area test
> - `/kind bug`
- [ ] /sig node
> /label kind/bug
`/remove-kind bug`"""
    assert prepare_text("Kubelet failure", body, labels) == (
        "Title: Kubelet failure\nBody: Keep this text."
    )


def test_preserves_head_and_tail_tokens() -> None:
    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(range(int(text))) for text in texts]}

        def build_inputs_with_special_tokens(self, token_ids):
            return [101, *token_ids, 102]

        def pad(self, rows, **kwargs):
            return rows[0]

    encoded = encode_head_tail(Tokenizer(), ["20"], 8)
    assert encoded["input_ids"] == [101, 0, 1, 2, 3, 18, 19, 102]
