from dataclasses import dataclass
import types

@dataclass
class Testando:
    texto: str = ""
    funcao: types.MethodType = lambda texto : texto

    def imprime(self):
        print(self.funcao(self.texto))

def emCaixaAlta(texto):
    return texto.upper()

def emCaixaBaixa(texto):
    return texto.lower()

def reverso(texto):
    return texto[::-1]

def main():
    test = Testando("Texto de Teste 1")
    test.imprime()

    test2 = Testando("Texto de Teste 2", funcao = emCaixaBaixa)
    test2.imprime()

    test2.funcao = emCaixaAlta
    test2.imprime()

    test2.funcao = reverso
    test2.imprime()
