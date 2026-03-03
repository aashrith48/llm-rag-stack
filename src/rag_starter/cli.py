import typer
from rich import print

from .ingest import ingest_path
from .rag import retrieve, build_prompt

app = typer.Typer(help="RAG Starter CLI")


@app.command()
def ingest(path: str):
    """Ingest a file or folder of .txt/.md into Chroma."""
    print(ingest_path(path))


@app.command()
def query(q: str, top_k: int = 5, prompt: bool = True):
    """Query the vector DB and optionally print a RAG prompt."""
    res = retrieve(q, top_k=top_k)
    print(res)
    if prompt:
        print("\n[bold]--- RAG Prompt ---[/bold]\n")
        print(build_prompt(q, res["hits"]))
