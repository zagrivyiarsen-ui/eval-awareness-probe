import logging
import pytest
from tests.fake_config import load_fake, smoke_model_name

log = logging.getLogger("conftest")


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    return load_fake(tmp_path_factory.mktemp("cfg_fake"))


@pytest.fixture(scope="session")
def tok(cfg):
    from src.prompts import load_tokenizer
    try:
        return load_tokenizer(cfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"plan tokenizer unavailable: {str(e)[:200]}")


@pytest.fixture(scope="session")
def smoke(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    from src.prompts import load_tokenizer
    name = smoke_model_name()
    try:
        t = load_tokenizer(cfg, name_override=name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        if torch.cuda.is_available():
            m = m.cuda()
        m.eval()
        log.info("SMOKE MODEL: %s dtype=%s device=%s", name, dtype, next(m.parameters()).device)
        return m, t, name
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"smoke model {name} unavailable: {str(e)[:200]}")
