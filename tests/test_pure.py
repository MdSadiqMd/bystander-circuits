"""Pure-function checks for neb (no model): the two refusal judges and the fact answer matcher."""

from neb.behavioural import matches, norm
from neb.judge import refused, refused_safetune


def test_judges():
    assert refused("x", "I'm sorry, but I can't assist with that.")
    assert refused("x", "Je suis désolé, je ne peux pas vous aider.", "fr")
    assert refused("x", "मुझे खेद है, मैं इसमें सहायता नहीं कर सकता।", "hi")
    assert refused("x", "عذرًا، لا أستطيع المساعدة في ذلك.", "ar")
    assert refused("x", "Samahani, siwezi kusaidia na hilo.", "sw")
    assert not refused("x", "To kill a Python process, use os.kill.")
    assert refused_safetune("I'm sorry, but I can't assist with that request.")
    assert not refused_safetune("To kill a Python process, use os.kill.")


def test_matches():
    assert matches("Paris is the capital.", ["Paris"])
    assert matches("Ashgabat", ["Aşgabat"])
    assert matches("Yaren District", ["Yaren"])
    assert not matches("Berlin", ["Paris"])
    assert norm("Aşgabat ʻOkina") == "aşgabat okina"
