def load():
    data = []
    targets = []

    f = open('sonar.data', 'r+')
    line = f.readline()
    while line:
        s = line.strip().split(',')
        d = s[0:len(s)-1]
        t = 1 if s[len(s)-1] == 'M' else 0

        for i in range(len(d)):
            d[i] = float(d[i])

        data.append(d)
        targets.append(t)

        line = f.readline()

    f.close()

    return data, targets
        
