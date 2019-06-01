import os
import random


def main():
    with open('numbers.txt', 'w') as f:
        for _ in range(100):
            f.write(str(random.randint(0, 100)) + os.linesep)


if __name__ == '__main__':
    main()
