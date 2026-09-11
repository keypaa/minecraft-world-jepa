from mw_jepa.action_tokenizer import parse_lumine_action


def test_noop_empty():
    assert parse_lumine_action("") == 28


def test_forward_key():
    s = "<|action_start|> 0 0 0 ; forward ;  ;  ;  <|action_end|>"
    assert parse_lumine_action(s) == 3


def test_attack_lmb():
    s = "<|action_start|> 0 0 0 ; LMB ;  ;  ;  <|action_end|>"
    assert parse_lumine_action(s) == 20
