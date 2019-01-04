
print('Prout', end=' ')
first = True
for i in range(0, 100000):
    if first:
        first = False
    else:
        print('\b' * len(str(i - 1)), end='')
    print(i, end='')
print('')
