counter = 0
with open("data/training_series_long.csv", "r+") as f :
    f.readline()
    lines = []
    line = f.readline()
    lines.append(line)
    last = line.split(',')
    while True :
        line = f.readline()
        counter+=1

        if not line:
            break
        values = line.split(',')
        if values[0] != last[0] or values[1] != last[1] :
            filename = "{0}-{1}.csv".format(last[0], last[1]).replace("/", "_")
            print("writing file", filename)
            print("processed", counter, "lines")
            with open("data/splits/"+filename, "w+") as f2:
                f2.writelines(lines)
                lines = []

        lines.append(line)
        last = line.split(',')