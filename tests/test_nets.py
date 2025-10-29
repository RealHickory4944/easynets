import easynets as en

def test_create_network():
  en.nets.create_network(num_layers=4,
                         layers=[2, 5, 5, 1], # input and output included
                         activation=['relu', 'relu', 'sigmoid'] # only hidden and output
