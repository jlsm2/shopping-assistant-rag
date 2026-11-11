from app.conversations import ConversationStore, Turn

FIRST = Turn("Tem set nude?", "Temos o Set Nude Curto.", product_ids=("nude-curto",))
SECOND = Turn("E quanto custa?", "R$ 59,90.", product_ids=("nude-curto",))


def store(clock=lambda: 0.0, **limits) -> ConversationStore:
    return ConversationStore(clock=clock, **limits)


def test_a_new_conversation_gets_an_unguessable_id_and_no_history():
    conversations = store()

    first_id, history = conversations.open(None, "ana")
    second_id, _ = conversations.open(None, "ana")

    assert history == ()
    assert len(first_id) >= 32 and first_id != second_id


def test_the_history_comes_back_in_order():
    conversations = store()
    conversation_id, _ = conversations.open(None, "ana")

    conversations.append(conversation_id, "ana", FIRST)
    conversations.append(conversation_id, "ana", SECOND)

    assert conversations.open(conversation_id, "ana") == (conversation_id, (FIRST, SECOND))


def test_someone_else_with_the_id_gets_a_new_empty_conversation():
    conversations = store()
    conversation_id, _ = conversations.open(None, "ana")
    conversations.append(conversation_id, "ana", FIRST)

    other_id, history = conversations.open(conversation_id, "bia")
    conversations.append(conversation_id, "bia", SECOND)  # cannot write into it either

    assert other_id != conversation_id and history == ()
    assert conversations.open(conversation_id, "ana")[1] == (FIRST,)


def test_an_unknown_id_starts_a_new_conversation():
    conversation_id, history = store().open("x" * 32, "ana")

    assert conversation_id != "x" * 32 and history == ()


def test_a_conversation_idle_past_the_ttl_is_forgotten():
    now = [0.0]
    conversations = store(clock=lambda: now[0], ttl_seconds=60)
    conversation_id, _ = conversations.open(None, "ana")
    conversations.append(conversation_id, "ana", FIRST)

    now[0] = 59.0
    assert conversations.open(conversation_id, "ana")[1] == (FIRST,)
    conversations.append(conversation_id, "ana", SECOND)  # activity renews it
    now[0] = 118.0
    assert conversations.open(conversation_id, "ana")[1] == (FIRST, SECOND)
    now[0] = 200.0
    assert conversations.open(conversation_id, "ana")[0] != conversation_id


def test_only_the_last_turns_are_kept():
    conversations = store(max_turns=2)
    conversation_id, _ = conversations.open(None, "ana")
    third = Turn("Tem vermelho?", "Temos o Set Vermelho Clássico.")

    for turn in (FIRST, SECOND, third):
        conversations.append(conversation_id, "ana", turn)

    assert conversations.open(conversation_id, "ana")[1] == (SECOND, third)


def test_the_least_recently_used_conversation_goes_first_when_full():
    now = [0.0]
    conversations = store(clock=lambda: now[0], max_conversations=2)
    ids = []
    for owner in ("ana", "bia", "cris"):
        now[0] += 1
        conversation_id, _ = conversations.open(None, owner)
        conversations.append(conversation_id, owner, FIRST)
        ids.append(conversation_id)

    assert conversations.open(ids[0], "ana")[1] == ()
    assert conversations.open(ids[2], "cris")[1] == (FIRST,)
