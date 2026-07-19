txt = "fhiusafsabfsakbfsabbbfsabfsabfsagsjsoigsf safsngfkjsn fafks iaflsanfkslfn"


def main():
    chunk = []

    j = 0

    s = ""
    for i in range(len(txt)):
        j += 1
        if j == 15:
            j = 0
            chunk.append(s)
            s = ""
        s += txt[i]
    print(chunk)


main()
