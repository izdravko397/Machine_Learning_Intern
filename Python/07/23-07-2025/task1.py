def leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# def days_between_dates(date1: str, date2: str):
#     MONTHS_OF_YEAR = 12
#     MONTHS_31_DAYS = {1, 3, 5, 7, 8, 10, 12}
#     MONTHS_30_DAYS = {4, 6, 9, 11}

#     day1, month1, year1 = (int(date) for date in date1.split('-'))
#     day2, month2, year2 = (int(date) for date in date2.split('-'))

#     if (year1, month1, day1) > (year2, month2, day2):
#         raise ValueError

#     res = 0
#     year_iterator = year1

#     if month1 == month2 and year1 == year2:
#         return day2 - day1

#     while year_iterator <= year2:
#         flag_start_days = False
#         flag_end_days = False

#         start_month = 1
#         end_month = MONTHS_OF_YEAR

#         if year_iterator == year1:
#             start_month = month1
#             flag_start_days = True

#         if year_iterator == year2:
#             end_month = month2
#             flag_end_days = True

#         for month in range(start_month, end_month + 1):
#             if month in MONTHS_30_DAYS:
#                 days = 30
#             elif month in MONTHS_31_DAYS:
#                 days = 31
#             else:
#                 days = 28
#                 if leap_year(year_iterator):
#                     days += 1
            
#             if flag_start_days and month == month1:
#                 days -= day1
#             elif flag_end_days and month == month2:
#                 days = day2

#             res += days                
#         year_iterator += 1

#     return res

def month_days(inx, year):
    february = 29 if leap_year(year) else 28
    list = [31, february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    return list[inx]

def days_between_dates(date1: str, date2: str):
    day1, month1, year1 = (int(date) for date in date1.split('-'))
    day2, month2, year2 = (int(date) for date in date2.split('-'))

    if (year1, month1, day1) > (year2, month2, day2):
        raise ValueError

    if year1 == year2:
        if month1 == month2:
            return day2 - day1
        
        # Diff Months
        first_month = month_days(month1 - 1, year1) - day1
        full_months = sum([month_days(m - 1, year1) for m in range(month1 + 1, month2)])
        last_month = day2
        return first_month + full_months + last_month
    
    # Diff years
    first_month = month_days(month1 - 1, year1) - day1
    days_to_year_end = sum([month_days(m, year1) for m in range(month1, 12)])
    
    days_between_years = 0
    for year in range(year1 + 1, year2):
        days_between_years += 365
        if leap_year(year):
            days_between_years += 1

    days_from_last_year = sum([month_days(m, year2) for m in range(month2 - 1)]) + day2

    return first_month + days_to_year_end + days_between_years + days_from_last_year


print(days_between_dates("11-06-2021", "12-06-2021"))
print(days_between_dates("1-01-2021", "1-02-2021"))
print(days_between_dates("1-01-2020", "1-01-2021"))
print(days_between_dates('01-01-2019', '01-01-2022')) #1096
print(days_between_dates('01-01-2025', '25-12-2024'))

# "11-06-2021", "12-06-2021" => 1
# "1-01-2021", "1-02-2021" => 31
