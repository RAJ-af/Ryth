import json
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from experiments.quantization import Int8Linear, compare, load_quantized, quantize_model, save_quantized
from experiments.retrieval import KNNMemory, evaluate
from model.checkpoint import save_checkpoint
from model.config import RythConfig
from model.decoder import RythForCausalLM


def tiny(tied=False):
    torch.manual_seed(42)
    return RythForCausalLM(RythConfig(vocab_size=16, d_model=8, n_layers=1,
        n_heads=2, n_kv_heads=1, ffn_hidden=16, max_seq_len=4,
        tie_embeddings=tied, use_flash_attention=False)).eval()


def test_memory_pairs_causality_and_leakage(tmp_path):
    model = tiny()
    docs = [[1, 2, 3, 4, 5, 6], [7, 8, 9]]
    memory = KNNMemory.build(model, docs, seq_len=4, max_entries=20)
    assert memory.values.tolist() == [2, 3, 4, 5, 6, 8, 9]
    with torch.no_grad():
        h, _ = model.decoder(torch.tensor([[1, 2, 3, 4]]))
        h_changed, _ = model.decoder(torch.tensor([[1, 2, 3, 9]]))
    torch.testing.assert_close(h[:, :3], h_changed[:, :3])
    torch.testing.assert_close(memory.keys[:4].float(), h[0].float(), atol=1e-3, rtol=1e-3)
    path = tmp_path / 'memory.pt'
    memory.save(path)
    loaded = KNNMemory.load(path)
    assert torch.equal(loaded.values, memory.values)
    with pytest.raises(ValueError, match="matches"):
        evaluate(model, loaded, docs)
    result = evaluate(model, loaded, [[1, 3, 5, 7, 9, 11]], seq_len=4, weight=0)
    assert result['tokens'] == 5
    assert result['model_ppl'] == pytest.approx(result['knn_ppl'])
    assert KNNMemory.build(model, docs, max_entries=2).values.tolist() == [2, 3]
    single = KNNMemory.build(model, [[1, 2]], max_entries=1)
    assert evaluate(model, single, [[1, 3]], weight=1, k=1)['knn_ppl'] == float('inf')


def test_search_interpolation_and_shapes():
    memory = KNNMemory(torch.tensor([[0., 0.], [1., 0.], [2., 0.]]),
                       torch.tensor([2, 2, 1]), 4)
    hidden = torch.tensor([[[0., 0.], [2., 0.]]])
    probs = memory.query(hidden, k=1, key_chunk=1, query_chunk=1)
    assert probs.argmax(-1).tolist() == [[2, 1]]
    logits = torch.zeros(1, 2, 4)
    mixed = memory.mix(logits, hidden, weight=.4, k=1)
    torch.testing.assert_close(mixed, .6 * torch.softmax(logits, -1) + .4 * probs)
    torch.testing.assert_close(memory.query(hidden, k=100).sum(-1), torch.ones(1, 2))
    assert memory.query(torch.empty(0, 2)).shape == (0, 4)
    with pytest.raises(ValueError):
        memory.query(hidden, k=0)


def test_int8_storage_matmul_tying_and_roundtrip(tmp_path):
    w = torch.tensor([[0., 0., 0.], [1., -2., .3]])
    layer = Int8Linear(w, torch.tensor([.2, -.1]))
    assert layer.qweight.dtype == torch.int8
    assert layer.qweight.element_size() == 1
    x = torch.randn(2, 4, 3)
    torch.testing.assert_close(layer(x), F.linear(x, w, layer.bias), atol=.02, rtol=.03)
    model = tiny()
    quantized = quantize_model(model)
    assert quantized.lm_head.qweight.dtype == torch.int8
    assert any(isinstance(m, Int8Linear) for m in quantized.decoder.modules())
    report = compare(model, quantized, [[1, 2, 3, 4]])
    assert report['int8_tensor_bytes'] < report['float_tensor_bytes']
    assert report['max_abs_logit_drift'] < .1
    path = tmp_path / 'quant.pt'
    save_quantized(path, quantized)
    restored = load_quantized(path)
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(quantized(ids)[0], restored(ids)[0])
    with pytest.raises(ValueError, match='tied'):
        quantize_model(tiny(tied=True))


def test_cli_checkpoint_end_to_end(tmp_path):
    model = tiny()
    ckpt = tmp_path / 'model.pt'
    save_checkpoint(str(ckpt), model, model.config)
    train, val = tmp_path / 'train.json', tmp_path / 'val.json'
    train.write_text(json.dumps([[1, 2, 3, 4, 5]]))
    val.write_text(json.dumps([[3, 4, 6, 7]]))
    mem = tmp_path / 'memory.pt'
    args = [sys.executable, '-m', 'experiments.retrieval', '--checkpoint', str(ckpt),
            '--datastore', str(train), '--eval', str(val), '--memory', str(mem)]
    result = json.loads(subprocess.check_output(args, text=True))
    assert result['tokens'] == 3
    assert json.loads(subprocess.check_output(args + ['--load-memory'], text=True)) == result
    quant = tmp_path / 'quant.pt'
    qargs = [sys.executable, '-m', 'experiments.quantization', '--checkpoint', str(ckpt),
             '--eval', str(val), '--output', str(quant)]
    assert json.loads(subprocess.check_output(qargs, text=True))['int8_tensor_bytes'] > 0
    assert quant.exists()
