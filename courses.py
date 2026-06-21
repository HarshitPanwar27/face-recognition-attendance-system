COURSES = {
    "ECE": {
        1: ["Mathematics-I", "Physics", "Basic Electronics", "English Communication", "Engineering Drawing"],
        2: ["Mathematics-II", "Chemistry", "Circuit Theory", "Programming in C", "Environmental Science"],
        3: ["Signals & Systems", "Electronic Devices", "Digital Electronics", "Network Theory", "Mathematics-III"],
        4: ["Analog Circuits", "Electromagnetic Theory", "Control Systems", "Microprocessors", "Probability & Statistics"],
        5: ["Digital Communication", "VLSI Design", "Microwave Engineering", "DSP", "Power Electronics"],
        6: ["Mobile Communication", "Analogue Electronics", "VLSI Design", "Embedded Systems", "Optical Fiber Communication"],
        7: ["Wireless Networks", "Advanced VLSI", "IoT Systems", "Satellite Communication", "Elective-I"],
        8: ["Project Work", "Industrial Training", "Seminar", "Elective-II", "Elective-III"],
    },
    "CSE": {
        1: ["Mathematics-I", "Physics", "Programming in C", "English Communication", "Engineering Drawing"],
        2: ["Mathematics-II", "Chemistry", "Data Structures", "Digital Logic", "Environmental Science"],
        3: ["Algorithms", "Discrete Mathematics", "OOP with Java", "Computer Organization", "Mathematics-III"],
        4: ["Operating Systems", "Database Management", "Software Engineering", "Theory of Computation", "Computer Networks-I"],
        5: ["Computer Networks-II", "Compiler Design", "Artificial Intelligence", "Web Technologies", "Probability & Statistics"],
        6: ["Cloud Computing", "Statistics & Probability", "Computer Networks", "Machine Learning", "Information Security"],
        7: ["Big Data Analytics", "Deep Learning", "Distributed Systems", "Elective-I", "Seminar"],
        8: ["Project Work", "Industrial Training", "Seminar", "Elective-II", "Elective-III"],
    },
}

# Fixed batches per department
DEPARTMENT_BATCHES = {
    "ECE": ["A1", "A2", "A3"],
    "CSE": ["B1", "B2", "B3"],
}

DEPARTMENTS = list(COURSES.keys())
SEMESTERS   = [1, 2, 3, 4, 5, 6, 7, 8]


def get_subjects(department: str, semester: int):
    dept = COURSES.get(department.upper())
    if not dept:
        return []
    return dept.get(int(semester), [])


def get_batches(department: str):
    return DEPARTMENT_BATCHES.get(department.upper(), [])