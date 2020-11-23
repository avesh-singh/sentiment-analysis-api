from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from model import *
from constants import *
from data import MAX_SENTENCE_LEN
from transformers import BertTokenizer
import pickle

model = Model(HIDDEN_SIZE, 1, N_LAYERS, LIN_DROPOUT, BIDIRECTIONAL).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def hello_world(request):
    text = request.POST['text']
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:MAX_SENTENCE_LEN-2]
    indexed = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
    tensor = torch.LongTensor(indexed).to(device)
    input_tensor = tensor.unsqueeze(0)
    sentiment = model(input_tensor)
    return Response('positive' if sentiment > 0 else 'negative')


if __name__ == '__main__':
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(view=hello_world, route_name='hello', request_method='POST')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()
