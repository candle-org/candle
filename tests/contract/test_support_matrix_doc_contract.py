from pathlib import Path


def test_support_matrix_exists():
    assert Path("docs/support-matrix.md").exists()


def test_readme_links_support_matrix():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "support-matrix" in readme.lower()
