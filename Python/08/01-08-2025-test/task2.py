def leap_years_days(year):
    return year // 4 - year // 100 + year // 400

def leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

list_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def month_days(month):
    sum = 0
    [sum := sum + list_months[i] for i in range(month - 1)]
    return sum

def day_of_week(date: str):
    week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day, month, year = (int(d) for d in date.split('-'))

    days_to_date = (year - 1) * 365

    days_to_date += leap_years_days(year - 1)

    days_to_date += month_days(month) + day

    if month > 2 and leap_year(year):
        days_to_date += 1

    return week[(days_to_date - 1) % 7]

# "30-11-2021" => 2 (вторник)
print(day_of_week('02-08-2025'))
print(day_of_week('01-01-0001'))