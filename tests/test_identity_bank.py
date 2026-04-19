from __future__ import annotations

from avatar_service.inference.identity_bank import IdentityBank, IdentityTokens


def test_put_get_evict_lru() -> None:
    bank = IdentityBank(capacity=3)
    for key in ("a", "b", "c"):
        bank.put(IdentityTokens(avatar_key=key, payload=object()))
    assert "a" in bank and "b" in bank and "c" in bank

    # Touch 'a' to refresh its LRU position, then insert 'd' and 'a' should survive, 'b' should evict.
    _ = bank.get("a")
    bank.put(IdentityTokens(avatar_key="d", payload=object()))

    assert "a" in bank
    assert "c" in bank
    assert "d" in bank
    assert "b" not in bank
    assert len(bank) == 3
