def netsummary(model) -> str:
    summ = []
    model.summary(print_fn=lambda x: summ.append(x))
    return '\n'.join(summ)