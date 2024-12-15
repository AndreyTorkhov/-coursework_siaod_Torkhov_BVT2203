import xlsxwriter
import time
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from deap import base, creator, tools, algorithms
from typing import List, Tuple, Dict, Any

class Bus:
    def __init__(self, bus_id: int, route, capacity: int = 40):
        self.bus_id = bus_id
        self.route = route
        self.schedule = []
        self.assigned_drivers = []
        self.current_passengers = 0
        self.capacity = capacity
        self.passengers = []
        self.total_boarded = 0
        self.total_alighted = 0


class Driver:
    class Shift:
        def __init__(self, work: tuple, rests: list):
            self.work = work
            self.rest = rests

    def __init__(self, driver_id: int, driver_type: int):
        self.driver_id = driver_id
        self.driver_type = driver_type
        self.shifts = []
        self.assigned_buses = []

    def add_shift(self, work: tuple, rests: list):
        shift = self.Shift(work, rests)
        self.shifts.append(shift)


class Route:
    def __init__(self, route_id: int, stops: list, average_time_between_stops: int, peak_variation: int, offpeak_variation: int):
        self.route_id = route_id
        self.stops = stops
        self.average_time_between_stops = average_time_between_stops
        self.peak_variation = peak_variation
        self.offpeak_variation = offpeak_variation
        self.schedule = []

        for stop in stops:
            stop.add_route(self)

    def add_schedule(self, trip_schedule: list):
        self.schedule.append(trip_schedule)

class Stop:
    def __init__(self, name: str, waiting_passengers: int = 0):
        self.name = name
        self.waiting_passengers = waiting_passengers
        self.routes = []

    def add_route(self, route):
        if route not in self.routes:
            self.routes.append(route)


def generate_stops(n: int) -> List[str]:
    return [f"BS{i+1}" for i in range(n)]

def time_to_minutes(time_str: str) -> int:
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

def minutes_to_time(minutes: int) -> str:
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def create_random_stops(stops_pool: List[str], min_stops: int, max_stops: int) -> List[str]:
    num_stops = random.randint(min_stops, max_stops)
    return random.sample(stops_pool, num_stops)

def create_route(route_id: int, stops: List[str], average_time_between_stops: int,
                 peak_variation: int, offpeak_variation: int):
    stop_objects = [Stop(name=stop) for stop in stops]
    return Route(
        route_id=route_id,
        stops=stop_objects,
        average_time_between_stops=average_time_between_stops,
        peak_variation=peak_variation,
        offpeak_variation=offpeak_variation
    )

def generate_random_routes(num_routes: int = 5,
                           stops_pool: List[str] = None,
                           stop_nums: int = 30,
                           min_stops: int = 5,
                           max_stops: int = 15,
                           average_time_between_stops: int = 5):
    if stops_pool is None:
        stops_pool = generate_stops(stop_nums)

    routes = []
    peak_variation = 2
    offpeak_variation = 5

    for route_id in range(1, num_routes + 1):
        selected_stops = create_random_stops(stops_pool, min_stops, max_stops)
        route = create_route(
            route_id=route_id,
            stops=selected_stops,
            average_time_between_stops=average_time_between_stops,
            peak_variation=peak_variation,
            offpeak_variation=offpeak_variation
        )
        routes.append(route)

    return routes

def is_in_peak_hours(current_time_min: int, peak_hours: Tuple[Tuple[int, int], ...]) -> bool:
    current_time_mod = current_time_min % 1440
    for peak_start, peak_end in peak_hours:
        peak_start_min = peak_start * 60
        peak_end_min = peak_end * 60
        if peak_start_min <= current_time_mod < peak_end_min:
            return True
    return False

def generate_trip_schedule(stops: List[Stop], start_time_min: int, average_time_between_stops: int,
                           variation_range: int) -> List[Tuple[str, str]]:
    trip_schedule = []
    trip_time = start_time_min
    for stop in stops:
        trip_schedule.append((stop.name, minutes_to_time(trip_time)))
        travel_time_variation = random.randint(-variation_range, variation_range)
        trip_time += average_time_between_stops + travel_time_variation
    return trip_schedule

def get_variation(route, current_time_min: int, peak_hours: Tuple[Tuple[int, int], ...]) -> int:
    if is_in_peak_hours(current_time_min, peak_hours):
        return random.randint(-route.peak_variation, route.peak_variation)
    else:
        return random.randint(-route.offpeak_variation, route.offpeak_variation)

def generate_route_schedule(route, start_time_min: int, operation_hours: int = 19, peak_hours=((7, 10), (17, 20))):
    schedule = []
    current_time_min = start_time_min
    end_time_min = start_time_min + operation_hours * 60

    forward_stops = route.stops
    backward_stops = list(reversed(route.stops))[1:-1]

    while current_time_min < end_time_min:
        trip_schedule_forward = generate_trip_schedule(
            forward_stops, current_time_min, route.average_time_between_stops, route.peak_variation
        )
        schedule.append(trip_schedule_forward)

        trip_schedule_backward = generate_trip_schedule(
            backward_stops, current_time_min + len(forward_stops) * route.average_time_between_stops + 10,
            route.average_time_between_stops, route.offpeak_variation
        )
        schedule.append(trip_schedule_backward)

        current_time_min += len(forward_stops) * route.average_time_between_stops * 2 + 20

    route.schedule = schedule
    return schedule

def manage_buses(routes, min_buses_per_route=2, max_wait_time=15, start_time=360):
    buses = []
    bus_id = 1

    for route in routes:
        avg_trip_time = len(route.stops) * route.average_time_between_stops * 2
        buffer_time = 10

        if avg_trip_time + buffer_time > 60:
            trips_per_hour = 1
        else:
            trips_per_hour = 60 // (avg_trip_time + buffer_time)

        max_trips_per_hour = 60 // max_wait_time
        required_buses = max(min_buses_per_route, (max_trips_per_hour // trips_per_hour) + 1)

        for i in range(2):
            bus = Bus(bus_id=bus_id, route=route)
            start_time_min = start_time + (i * 5)
            generate_route_schedule(route, start_time_min)
            bus.schedule = copy.deepcopy(route.schedule)
            buses.append(bus)
            bus_id += 1

        interval_between_buses = (1440 // required_buses) if required_buses > 2 else 0
        for i in range(required_buses - 2):
            bus = Bus(bus_id=bus_id, route=route)
            start_time_min = start_time + (i * interval_between_buses)
            generate_route_schedule(route, start_time_min)
            bus.schedule = copy.deepcopy(route.schedule)
            buses.append(bus)
            bus_id += 1

    return buses

def is_trip_within_shift(trip_start: int, trip_end: int, shift_start: int, shift_end: int) -> bool:
    return trip_start >= shift_start and trip_end <= shift_end

def has_rest_conflict(trip_start: int, trip_end: int, rest_periods: List[Tuple[int, int]]) -> bool:
    return any(trip_start < rest_end and trip_end > rest_start for rest_start, rest_end in rest_periods)

def can_assign(driver: 'Driver', bus: 'Bus') -> bool:
    bus_trips = [(time_to_minutes(trip[0][1]), time_to_minutes(trip[-1][1])) for trip in bus.schedule]

    for trip_bus_start, trip_bus_end in bus_trips:
        can_work_this_trip = False
        for shift in driver.shifts:
            if is_trip_within_shift(trip_bus_start, trip_bus_end, shift.work[0], shift.work[1]) \
               and not has_rest_conflict(trip_bus_start, trip_bus_end, shift.rest):
                can_work_this_trip = True
                break
        if not can_work_this_trip:
            return False

    return True

def generate_driver_schedule(driver_type: int) -> Tuple[int, int, List[Tuple[int, int]]]:
    if driver_type == 1:
        work_start = 480   # 08:00
        work_end = 960     # 16:00
        rest_periods = [(780, 840)]  # Обед 13:00-14:00
    elif driver_type == 2:
        work_start = 480   # 08:00
        work_end = 1200    # 20:00
        rest_periods = [(600, 610), (900, 910)]  # Короткие перерывы в 10:00-10:10 и 15:00-15:10
    else:
        work_start = 480
        work_end = 960
        rest_periods = [(780, 840)]

    return work_start, work_end, rest_periods

def generate_passengers(stops: List['Stop'], min_passengers: int = 2, max_passengers: int = 5) -> None:
    for stop in stops:
        new_passengers = random.randint(min_passengers, max_passengers)
        stop.waiting_passengers += new_passengers


def process_bus_stop(bus: 'Bus', stop: 'Stop', current_time_min: int, history_log: List[Dict[str, Any]],
                     ticket_price: int = 50) -> None:
    passengers_to_alight = [p for p in bus.passengers if p[0] == stop.name]
    num_alighting = len(passengers_to_alight)
    bus.passengers = [p for p in bus.passengers if p[0] != stop.name]
    bus.current_passengers -= num_alighting
    bus.total_alighted += num_alighting

    available_seats = bus.capacity - bus.current_passengers
    passengers_waiting = stop.waiting_passengers

    boarding_passengers = min(passengers_waiting, available_seats)
    stop.waiting_passengers -= boarding_passengers
    bus.current_passengers += boarding_passengers
    bus.total_boarded += boarding_passengers

    for _ in range(boarding_passengers):
        possible_destinations = [s.name for s in bus.route.stops if s.name != stop.name]
        destination = random.choice(possible_destinations)
        bus.passengers.append((destination, current_time_min))

    earnings = boarding_passengers * ticket_price

    history_log.append({
        'Time': minutes_to_time(current_time_min),
        'Bus_ID': bus.bus_id,
        'Route_ID': bus.route.route_id,
        'Stop': stop.name,
        'On_Stop': passengers_waiting,
        'Boarded': boarding_passengers,
        'Alighted': num_alighting,
        'Current_Passengers': bus.current_passengers,
        'Earnings': earnings
    })

def process_bus_trips(buses: List['Bus'], history_log: List[Dict[str, Any]], ticket_price: int = 50) -> None:
    for bus in buses:
        for trip in bus.schedule:
            for stop_name, arrival_time_str in trip:
                arrival_time_min = time_to_minutes(arrival_time_str)
                stop = next((s for s in bus.route.stops if s.name == stop_name), None)
                if stop is None:
                    continue
                process_bus_stop(bus, stop, arrival_time_min, history_log, ticket_price)

def simulate_day(buses, stops, history_log, drivers, ticket_price=50, driver_salary=2000):

    for bus in buses:
        bus.current_passengers = 0
        bus.passengers = []
        bus.total_boarded = 0
        bus.total_alighted = 0

    for stop in stops:
        stop.waiting_passengers = 0

    SIMULATION_START = 6 * 60
    SIMULATION_END = 25 * 60
    TIME_INTERVAL = 15

    event_schedule = defaultdict(list)
    for bus in buses:
        for trip in bus.schedule:
            for stop_name, arrival_time_str in trip:
                arrival_time_min = time_to_minutes(arrival_time_str)
                event_schedule[arrival_time_min].append((bus, stop_name))

    for current_time in range(SIMULATION_START, SIMULATION_END, TIME_INTERVAL):
        is_peak = 7 * 60 <= current_time < 10 * 60 or 17 * 60 <= current_time < 20 * 60
        generate_passengers(stops, min_passengers=2 if is_peak else 3, max_passengers=5 if is_peak else 6)

        overcrowded_stops = [stop for stop in stops if stop.waiting_passengers > 50]
        for stop in overcrowded_stops:
            for bus in buses:
                if stop in bus.route.stops:
                    new_bus_id = max(bus.bus_id for bus in buses) + 1
                    new_bus = Bus(bus_id=new_bus_id, route=bus.route)

                    start_time_min = current_time + 5
                    new_schedule = generate_route_schedule(bus.route, start_time_min)
                    new_bus.schedule = new_schedule

                    buses.append(new_bus)
                    event_schedule[start_time_min].append((new_bus, stop.name))

                    break

        if current_time in event_schedule:
            for bus, stop_name in event_schedule[current_time]:
                stop = next((s for s in stops if s.name == stop_name), None)
                if not stop:
                    continue
                process_bus_stop(bus, stop, current_time, history_log, ticket_price)

    total_boarded = sum(bus.total_boarded for bus in buses)
    income = total_boarded * ticket_price
    expenses = len(drivers) * driver_salary
    profit = income - expenses

    total_leftover_passengers = sum(stop.waiting_passengers for stop in stops)

    df_history = pd.DataFrame(history_log)
    summary = {
        'Total_Boarded': total_boarded,
        'Income': income,
        'Expenses': expenses,
        'Profit': profit,
        'Total_Leftover_Passengers': total_leftover_passengers
    }

    return df_history, summary


def assign_drivers_greedy(buses: List['Bus'], initial_driver_count: int = 10) -> List['Driver']:
    drivers = []
    driver_id = 1

    for _ in range(initial_driver_count):
        driver_type = random.choice([1, 2])
        driver = Driver(driver_id=driver_id, driver_type=driver_type)
        work_start, work_end, rest_periods = generate_driver_schedule(driver_type)
        driver.add_shift((work_start, work_end), rest_periods)
        drivers.append(driver)
        driver_id += 1

    buses_sorted = sorted(buses, key=lambda b: time_to_minutes(b.schedule[0][0][1]))
    driver_assignments = defaultdict(list)

    for bus in buses_sorted:
        assigned = False
        possible_drivers = [d for d in drivers if can_assign(d, bus)]

        if possible_drivers:
            best_driver = min(possible_drivers, key=lambda d: len(d.assigned_buses))
            best_driver.assigned_buses.append(bus)
            bus.assigned_drivers.append(best_driver.driver_id)
            driver_assignments[best_driver.driver_id].append(bus)
            assigned = True

        if not assigned:
            new_driver = Driver(driver_id=driver_id, driver_type=random.choice([1, 2]))
            work_start, work_end, rest_periods = generate_driver_schedule(new_driver.driver_type)
            new_driver.add_shift((work_start, work_end), rest_periods)

            new_driver.assigned_buses.append(bus)
            bus.assigned_drivers.append(new_driver.driver_id)
            drivers.append(new_driver)
            driver_assignments[new_driver.driver_id].append(bus)
            driver_id += 1

    def optimize_driver_assignments():
        for i in range(len(buses_sorted)):
            for j in range(i + 1, len(buses_sorted)):
                bus1 = buses_sorted[i]
                bus2 = buses_sorted[j]

                for d_id in bus1.assigned_drivers.copy():
                    driver1 = next(d for d in drivers if d.driver_id == d_id)

                    if d_id in bus2.assigned_drivers:
                        continue

                    if can_assign(driver1, bus2):
                        driver1.assigned_buses.remove(bus1)
                        driver1.assigned_buses.append(bus2)
                        bus1.assigned_drivers.remove(d_id)
                        bus2.assigned_drivers.append(d_id)
    optimize_driver_assignments()

    return drivers

def genetic_driver_assignment(buses: List['Bus'],
                              population_size: int = 100,
                              generations: int = 100,
                              cxpb: float = 0.7,
                              mutpb: float = 0.2) -> List['Driver']:

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_driver", random.randint, 0, len(buses) - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_driver, n=len(buses))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        driver_assignments = defaultdict(list)

        for bus_idx, driver_idx in enumerate(individual):
            driver_assignments[driver_idx].append(buses[bus_idx])

        num_drivers = len(driver_assignments)

        for assigned_buses in driver_assignments.values():
            driver_type = random.choice([1, 2])
            work_start, work_end, rest_periods = generate_driver_schedule(driver_type)

            for bus in assigned_buses:
                for trip in bus.schedule:
                    trip_start_min = time_to_minutes(trip[0][1])
                    trip_end_min = time_to_minutes(trip[-1][1])
                    if not (work_start <= trip_start_min and trip_end_min <= work_end):
                        return (float('inf'),)

        return (num_drivers,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(buses) - 1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))

    algorithms.eaSimple(population, toolbox, cxpb, mutpb, generations, stats=stats, halloffame=hof, verbose=False)

    best_ind = hof[0]

    driver_assignments = defaultdict(list)
    for bus_idx, driver_idx in enumerate(best_ind):
        driver_assignments[driver_idx].append(buses[bus_idx])

    drivers = []
    driver_id = 1
    for assigned_buses in driver_assignments.values():
        driver_type = random.choice([1, 2])
        driver = Driver(driver_id=driver_id, driver_type=driver_type)
        work_start, work_end, rest_periods = generate_driver_schedule(driver_type)
        driver.add_shift((work_start, work_end), rest_periods)
        driver.assigned_buses = assigned_buses
        drivers.append(driver)
        driver_id += 1

    return drivers

def export_to_excel(drivers: List['Driver'], routes: List['Route'], buses: List['Bus'],
                    history_log: List[Dict[str, Any]], summary: Dict[str, Any], filename: str = "schedule.xlsx"):

    workbook = xlsxwriter.Workbook(filename)

    header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'bg_color': '#D9EAD3'})
    cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
    time_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1, 'num_format': 'hh:mm'})
    money_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1, 'num_format': '#,##0 ₽'})
    number_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1, 'num_format': '#,##0'})


    def write_sheet(sheet_name: str, headers: List[str], rows: List[List[Any]], formats: List[Any] = None):
        sheet = workbook.add_worksheet(sheet_name)
        for col, header in enumerate(headers):
            sheet.write(0, col, header, header_format)

        for row_num, row in enumerate(rows, start=1):
            for col_num, value in enumerate(row):
                fmt = formats[col_num] if formats and col_num < len(formats) else cell_format
                sheet.write(row_num, col_num, value, fmt)

    # Лист "Водители"
    driver_rows = [
        [
            driver.driver_id,
            driver.driver_type,
            ", ".join(map(str, [bus.bus_id for bus in driver.assigned_buses])),
            ", ".join([f"{minutes_to_time(shift.work[0])}-{minutes_to_time(shift.work[1])}" for shift in driver.shifts]),
            ", ".join([f"{minutes_to_time(rest[0])}-{minutes_to_time(rest[1])}" for shift in driver.shifts for rest in shift.rest])
        ]
        for driver in drivers
    ]
    write_sheet("Водители", ["ID водителя", "Тип водителя", "Назначенные автобусы", "График работы", "График отдыха"], driver_rows)

    # Лист "Маршруты"
    route_rows = [
        [
            route.route_id,
            ", ".join([stop.name for stop in route.stops]),
            "\n".join([", ".join([f"{stop} {time}" for stop, time in trip]) for trip in route.schedule]),
            ", ".join(map(str, [bus.bus_id for bus in buses if bus.route.route_id == route.route_id]))
        ]
        for route in routes
    ]
    write_sheet("Маршруты", ["ID маршрута", "Остановки", "Расписание", "Назначенные автобусы"], route_rows)

    # Лист "Автобусы"
    bus_rows = [
        [
            bus.bus_id,
            bus.route.route_id,
            ", ".join(map(str, bus.assigned_drivers)),
            "\n".join([", ".join([f"{stop} {time}" for stop, time in trip]) for trip in bus.schedule])
        ]
        for bus in buses
    ]
    write_sheet("Автобусы", ["ID автобуса", "ID маршрута", "Назначенные водители", "Расписание"], bus_rows)

    # Лист "Остановки"
    stop_rows = []
    for bus in buses:
        for trip in bus.schedule:
            for stop_name, time in trip:
                stop_rows.append([
                    bus.route.route_id,
                    stop_name,
                    bus.bus_id,
                    time
                ])
    write_sheet("Остановки", ["ID маршрута", "Остановка", "ID автобуса", "Время прибытия"], stop_rows, formats=[cell_format, cell_format, cell_format, time_format])

    # Лист "Хронология дня"
    chronology_rows = [
        [
            log['Time'], log['Bus_ID'], log['Route_ID'], log['Stop'],
            log['On_Stop'], log['Boarded'], log['Alighted'],
            log['Current_Passengers'], log['Earnings']
        ]
        for log in history_log
    ]
    write_sheet(
        "Хронология дня",
        ['Время', 'ID автобуса', 'ID маршрута', 'Остановка', 'На остановке',
         'Посадка', 'Высадка', 'Текущие пассажиры', 'Заработок (₽)'],
        chronology_rows,
        formats=[cell_format] * 8 + [money_format]
    )

     # Лист "Финансы"
    financial_rows = [
        ["Общее число пассажиров", summary['Total_Boarded']],
        ["Общий заработок (₽)", summary['Income']],
        ["Расходы (₽)", summary['Expenses']],
        ["Прибыль (₽)", summary['Profit']],
    ]

    financial_formats = [cell_format, number_format, money_format, money_format, money_format]

    write_sheet(
        "Финансы",
        ["Показатель", "Значение"],
        financial_rows,
        formats=[cell_format, number_format]
    )

    workbook.close()

def plot_comparison(
    experiments: List[int], values_1: List[float], values_2: List[float], labels: Dict[str, str],
    y_label: str, title: str, output_file: str, value_offsets: Dict[str, float] = None
):
    plt.figure(figsize=(12, 8))
    plt.plot(experiments, values_1, marker='o', color='blue', label=labels['line1'])
    plt.plot(experiments, values_2, marker='o', color='green', label=labels['line2'])

    plt.xlabel('Эксперимент', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(experiments, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)

    offsets = value_offsets or {'line1': 0.05, 'line2': 0.05}
    for i in range(len(experiments)):
        plt.text(experiments[i], values_1[i] + offsets['line1'], f'{values_1[i]:,.0f}',
                 ha='center', va='bottom', fontsize=10, color='blue')
        plt.text(experiments[i], values_2[i] + offsets['line2'], f'{values_2[i]:,.0f}',
                 ha='center', va='bottom', fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def compare_experiments(experiment_data: List[Dict[str, Any]]) -> None:

    if not experiment_data:
        print("Нет данных для сравнения экспериментов.")
        return

    greedy_times = [data['greedy_time'] for data in experiment_data]
    genetic_times = [data['genetic_time'] for data in experiment_data]
    greedy_drivers = [data['greedy_drivers'] for data in experiment_data]
    genetic_drivers = [data['genetic_drivers'] for data in experiment_data]
    greedy_profit = [data['greedy_profit'] for data in experiment_data]
    genetic_profit = [data['genetic_profit'] for data in experiment_data]

    experiments = list(range(1, len(experiment_data) + 1))

    plot_comparison(
        experiments=experiments,
        values_1=greedy_times,
        values_2=genetic_times,
        labels={'line1': 'Жадный алгоритм', 'line2': 'Генетический алгоритм'},
        y_label='Время выполнения (сек)',
        title='Сравнение времени выполнения алгоритмов',
        output_file='result/execution_time_comparison.png',
        value_offsets={'line1': 0.01, 'line2': 0.01}
    )

    plot_comparison(
        experiments=experiments,
        values_1=greedy_drivers,
        values_2=genetic_drivers,
        labels={'line1': 'Жадный алгоритм', 'line2': 'Генетический алгоритм'},
        y_label='Количество водителей',
        title='Сравнение количества водителей',
        output_file='result/drivers_count_comparison.png',
        value_offsets={'line1': 0.1, 'line2': 0.1}
    )

    plot_comparison(
        experiments=experiments,
        values_1=greedy_profit,
        values_2=genetic_profit,
        labels={'line1': 'Жадный алгоритм', 'line2': 'Генетический алгоритм'},
        y_label='Прибыль (руб.)',
        title='Сравнение прибыли алгоритмов',
        output_file='result/profit_comparison.png',
        value_offsets={'line1': 100, 'line2': 100}
    )

def main():
    experiment_data = []

    num_routes = 5
    min_buses_per_route = 3
    initial_driver_count = 15
    stop_nums = 30

    for experiment_num in range(1, 6):
        print(f"\n--- Эксперимент {experiment_num} ---")

        stops = [Stop(name) for name in generate_stops(stop_nums)]
        routes = generate_random_routes(
            num_routes=num_routes,
            stops_pool=[stop.name for stop in stops],
            min_stops=5,
            max_stops=10
        )

        buses = manage_buses(routes, min_buses_per_route=min_buses_per_route)

        start_greedy = time.time()
        drivers_greedy = assign_drivers_greedy(buses, initial_driver_count=initial_driver_count)
        greedy_time = time.time() - start_greedy

        history_log_greedy = []
        df_history_greedy, summary_greedy = simulate_day(
            buses, stops, history_log_greedy, drivers_greedy
        )

        export_to_excel(
            drivers_greedy,
            routes,
            buses,
            history_log_greedy,
            summary_greedy,
            filename=f"schedule_greedy_{experiment_num}.xlsx"
        )


        buses_copy = copy.deepcopy(buses)
        start_genetic = time.time()
        drivers_genetic = genetic_driver_assignment(
            buses_copy,
            population_size=100,
            generations=100,
            cxpb=0.7,
            mutpb=0.2
        )
        genetic_time = time.time() - start_genetic

        history_log_genetic = []
        df_history_genetic, summary_genetic = simulate_day(
            buses_copy, stops, history_log_genetic, drivers_genetic
        )

        export_to_excel(
            drivers_genetic,
            routes,
            buses_copy,
            history_log_genetic,
            summary_genetic,
            filename=f"schedule_genetic_{experiment_num}.xlsx"
        )

        experiment_data.append({
            'greedy_time': greedy_time,
            'genetic_time': genetic_time,
            'greedy_drivers': len(drivers_greedy),
            'genetic_drivers': len(drivers_genetic),
            'greedy_profit': summary_greedy['Profit'],
            'genetic_profit': summary_genetic['Profit']
        })

        print(f"Жадный алгоритм: Время = {greedy_time:.4f} сек, Водители = {len(drivers_greedy)}, Прибыль = {summary_greedy['Profit']} руб.")
        print(f"Генетический алгоритм: Время = {genetic_time:.4f} сек, Водители = {len(drivers_genetic)}, Прибыль = {summary_genetic['Profit']} руб.")

    compare_experiments(experiment_data)

if __name__ == "__main__":
    main()

