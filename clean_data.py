import csv

names = dict()
with open("data.csv", "r") as csvfile, open("data_clean.csv", "w") as csvcleanfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    writer = csv.writer(csvcleanfile, delimiter=',', quotechar='"')
    for row in reader:
        if row == []:
            continue
        if row.count("---") <= 2:
            if not names.get(row[1]):
                names[row[1]] = row
            else:
                if row.count("---") < names[row[1]]:
                    names[row[1]] = row

    for name in names:
        writer.writerow(names[name])
