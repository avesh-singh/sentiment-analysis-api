from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from model import model
from dataset import Tokenizer
from constants import *
import torch

tokenizer = Tokenizer()
model = model.to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()


def sentiment(request):
    text = request.POST['text']
    tokenized = tokenizer(text, MAX_SENTENCE_LEN)
    output = model(tokenized['input_ids'].to(device), tokenized['attention_mask'].to(device))
    prediction = torch.max(output, dim=1)[1]
    resp = {'sentiment': 'positive' if prediction > 0 else 'negative'}
    return resp


if __name__ == '__main__':
    with Configurator() as config:
        config.add_route('sentiment', '/sentiment')
        config.add_view(sentiment, route_name='sentiment', request_method='POST', renderer='json')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()
