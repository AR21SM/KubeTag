from kubetag.text_processing import prepare_text

def test_prepare_text_basic() -> None:
    title = "sig-auth: token request failing"
    body = "Getting unauthorized errors"
    result = prepare_text(title, body)
    assert result == "Title: sig-auth: token request failing\nBody: Getting unauthorized errors"

def test_prepare_text_null_body() -> None:
    result = prepare_text("simple title", None)
    assert result == "Title: simple title\nBody:"

def test_prepare_text_command_leakage_removal() -> None:
    title = "Fix auth issue"
    body = """
    /assign @someone
    /close
    This is actual descriptive text.
    /sig auth
    """
    result = prepare_text(title, body)
    assert result == "Title: Fix auth issue\nBody: This is actual descriptive text."

def test_prepare_text_taxonomy_token_removal() -> None:
    title = "Bug with kind/bug and sig/auth labels"
    body = "Checking area/kubectl issues"
    result = prepare_text(title, body)
    assert result == "Title: Bug with and labels\nBody: Checking issues"

def test_prepare_text_whitespace_normalization() -> None:
    title = "  Too    many   spaces   "
    body = "Line 1\n\n\nLine 2"
    result = prepare_text(title, body)
    assert result == "Title: Too many spaces\nBody: Line 1\nLine 2"

def test_prepare_text_truncation() -> None:
    title = "title"
    body = "x" * 2500
    result = prepare_text(title, body)
    assert len(result) == 2000
