import torch

from thinker.core.rnn import MaskedGRU


def test_masked_gru_freezes_wait_rows_and_their_input_gradient():
    torch.manual_seed(7)
    encoder = MaskedGRU(input_size=3, hidden_size=5)
    tokens = torch.randn(3, 2, 3, requires_grad=True)
    update_mask = torch.tensor(
        [[True, True], [False, True], [True, False]], dtype=torch.bool
    )

    output, final_state = encoder(tokens, update_mask=update_mask)

    assert torch.equal(output[1, 0], output[0, 0])
    assert torch.equal(output[2, 1], output[1, 1])
    assert torch.equal(final_state[0][1], output[1, 1])

    output[-1].sum().backward()
    assert torch.count_nonzero(tokens.grad[1, 0]) == 0
    assert torch.count_nonzero(tokens.grad[2, 1]) == 0


def test_masked_gru_resets_before_consuming_a_root_token():
    torch.manual_seed(11)
    encoder = MaskedGRU(input_size=3, hidden_size=5)
    tokens = torch.randn(3, 1, 3)
    reset_mask = torch.tensor([[False], [False], [True]])

    output, _ = encoder(tokens, reset_mask=reset_mask)
    root_only, _ = encoder(tokens[2:3], state=encoder.initial_state(1))

    torch.testing.assert_close(output[2], root_only[0], rtol=0, atol=0)
