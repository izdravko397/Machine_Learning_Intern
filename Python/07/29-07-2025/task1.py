def leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def month_days(start_inx, end_index, year):
    february = 29 if leap_year(year) else 28
    list = [31, february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    return list[start_inx:end_index]

def days_first_month(month, from_day, year): 
    return month_days(month - 1, month, year)[0] - from_day

def days_between_dates(date1: str, date2: str):
    day1, month1, year1 = (int(date) for date in date1.split('-'))
    day2, month2, year2 = (int(date) for date in date2.split('-'))

    if (year1, month1, day1) > (year2, month2, day2):
        raise ValueError

    if year1 == year2:
        if month1 == month2:
            return day2 - day1
        
        # Diff Months
        first_month = days_first_month(month1, day1, year1)
        full_months = sum(month_days(month1, month2 - 1, year1))
        last_month = day2
        return first_month + full_months + last_month
    
    # Diff years
    first_month = days_first_month(month1, day1, year1)
    days_to_year_end = sum(month_days(month1, 12, year1))
    
    days_between_years = 0
    for year in range(year1 + 1, year2):
        days_between_years += 365
        if leap_year(year):
            days_between_years += 1

    days_from_last_year = sum(month_days(0, month2 - 1, year2)) + day2

    return first_month + days_to_year_end + days_between_years + days_from_last_year


print(days_between_dates("11-06-2021", "12-06-2021")) # 1
print(days_between_dates("1-01-2021", "1-02-2021"))  # 31
print(days_between_dates("1-01-2020", "1-01-2021")) # 366 visokosna
print(days_between_dates("1-01-2025", "1-01-2026")) # 365
print(days_between_dates('01-01-2019', '01-01-2022')) #1096
print(days_between_dates('01-01-2025', '25-12-2024')) # invalid input

# "11-06-2021", "12-06-2021" => 1
# "1-01-2021", "1-02-2021" => 31