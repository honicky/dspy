import pytest
from pytest_mock.plugin import MockerFixture
import requests
from transformers import AutoModelForSeq2SeqLM

import dspy
from dsp.modules.hf_client import Together

class MockConfig:
    def __init__(self, architectures: list[str]):
        self.architectures = architectures


def test_load_gated_model(mocker: MockerFixture):
    conf = MockConfig(architectures=["ConditionalGeneration"])
    mocker.patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    mocker.patch("transformers.AutoConfig.from_pretrained", return_value=conf)
    mocker.patch("transformers.AutoTokenizer.from_pretrained")

    some_token = "asdfasdfasdf"
    model = "google/gemma-7b"
    _ = dspy.HFModel(model, token=some_token)
    AutoModelForSeq2SeqLM.from_pretrained.assert_called_with(model, device_map="auto", token=some_token)


def test_load_ungated_model(mocker: MockerFixture):
    conf = MockConfig(architectures=["ConditionalGeneration"])
    mocker.patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    mocker.patch("transformers.AutoConfig.from_pretrained", return_value=conf)
    mocker.patch("transformers.AutoTokenizer.from_pretrained")
    _ = dspy.HFModel("openai-community/gpt2")
    # no token used in automodel
    AutoModelForSeq2SeqLM.from_pretrained.assert_called_with("openai-community/gpt2", device_map="auto", token=None)


@pytest.fixture
def together_client(mocker: MockerFixture):
    """
    Mock the requests.Session and the response from the Together API.

    We remove the backoff decorator from _generate method so that we can avoid waiting for the backoff time
    in the case where we get an exception, either expected or unexpected.
    """

    mock_session = mocker.patch("requests.Session")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"text": "some text"}]}
    mock_session.return_value.post.return_value = mock_response

    client = Together("some-model")
    client._generate = client._generate.__wrapped__.__get__(client, Together) # remove the decorator

    return client


def test_together_client_happy_path(together_client):

    assert together_client.basic_request("some prompt") == {"prompt": "some prompt", "choices": [{"text": "some text"}]}


def test_together_client_bad_status(together_client: Together):

    together_client.session.post.return_value.status_code = 401

    with pytest.raises(Exception):
        together_client.basic_request("some prompt")


def test_together_client_bad_json(together_client):

    together_client.session.post.return_value.json.side_effect = requests.exceptions.JSONDecodeError("some error", "some doc", 0)

    with pytest.raises(Exception):
        together_client.basic_request("some prompt")
