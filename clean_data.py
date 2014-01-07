import csv

names = dict()
with open("data.csv", "r") as csvfile, open("data_clean.csv", "w") as csvcleanfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    writer = csv.writer(csvcleanfile, delimiter=',', quotechar='"')
    for row in reader:
        if row == []:
            continue
        if row.count("---") <= 2:
            if not names.get(row[2]):
                names[row[2]] = row
            else:
                if row.count("---") < names[row[2]]:
                    names[row[2]] = row


    order = ["/sets/", "/setsTD/", "/good/", "/nonbenzo/"]
    for name in names:
        num = -1
        for i, x in enumerate(order):
            if x in names[name][0]:
                num = i
        writer.writerow([num] + names[name])
