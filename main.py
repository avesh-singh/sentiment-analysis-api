from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from model import *
from constants import *
from data import convert_to_seqs
import pickle

model = Model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 1, DROPOUT, N_LAYERS).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


def hello_world(request):
    print()
    text = [request.POST['text']]
    input_tensor = convert_to_seqs(tokenizer, text)
    print(input_tensor)
    sentiment = model(input_tensor)
    print(sentiment)
    return Response('positive' if sentiment > 0 else 'negative')


if __name__ == '__main__':
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(view=hello_world, route_name='hello', request_method='POST')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()
