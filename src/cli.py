import fire


def RagCli():
    pass


def greet(name="World", uppercase=False):
    text = f"Hello, {name}!"
    if uppercase:
        return text.upper()
    return text


if __name__ == "__main__":
    fire.Fire(greet)
