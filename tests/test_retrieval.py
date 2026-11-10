import pytest

from app.retrieval import Chunk, Index, chunk_markdown, load_chunks, slugify, stem, tokenize


def test_tokenize_strips_accents_and_lowercases():
    assert tokenize("DEVOLUÇÃO") == tokenize("devolucao")


def test_tokenize_removes_portuguese_stopwords():
    assert tokenize("qual o prazo para a troca") == ["praz", "troc"]


def test_tokenize_keeps_numbers_and_drops_single_letters():
    assert tokenize("R$ 150 em 7 dias") == ["150", "7", "dia"]


@pytest.mark.parametrize(
    "variants",
    [
        ("troca", "trocar", "trocas", "trocado"),
        ("pagamento", "pagar", "pago"),
        ("cupom", "cupons"),
        ("preto", "preta", "pretas"),
        ("entrega", "entregar", "entregamos"),
    ],
)
def test_stem_groups_word_variants(variants):
    assert len({stem(word) for word in variants}) == 1


def test_stem_never_shortens_below_three_letters():
    assert stem("uso") == "uso"


def test_chunk_markdown_makes_one_chunk_per_section_with_stable_ids():
    markdown = "# Título\nintro ignorada\n## Prazo para troca\n7 dias.\n\n## Frete grátis\nAcima."

    chunks = chunk_markdown("trocas-e-devolucoes", markdown)

    assert chunks == [
        Chunk("trocas-e-devolucoes#prazo-para-troca", "Prazo para troca", "7 dias."),
        Chunk("trocas-e-devolucoes#frete-gratis", "Frete grátis", "Acima."),
    ]


def test_slugify_is_ascii_and_hyphenated():
    assert slugify("  Devolução: por arrependimento! ") == "devolucao-por-arrependimento"


def test_load_chunks_rejects_duplicate_ids(tmp_path):
    (tmp_path / "faq.md").write_text("## Frete\na\n## Frete\nb\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate chunk ids"):
        load_chunks(tmp_path)


def test_search_ranks_title_match_above_body_match():
    in_title = Chunk("a#1", "Reembolso", "Texto neutro sobre outra coisa qualquer aqui.")
    in_body = Chunk("a#2", "Outro assunto", "Texto neutro que cita reembolso uma vez aqui.")
    filler = [Chunk(f"f#{n}", f"Tema {n}", "Conteúdo sem relação nenhuma.") for n in range(5)]

    results = Index([in_body, in_title, *filler]).search("reembolso")

    assert [chunk.id for chunk in results] == ["a#1", "a#2"]


def test_search_matches_query_without_accents(index):
    results = index.search("devolucao por arrependimento")

    assert results[0].id == "trocas-e-devolucoes#devolucao-por-arrependimento"


def test_search_returns_nothing_below_minimum_score(index):
    assert index.search("Qual a capital da Austrália?") == []


def test_search_respects_top_k(index):
    assert len(index.search("frete prazo entrega pedido troca", k=2)) == 2


def test_empty_index_returns_nothing():
    assert Index([]).search("frete") == []
