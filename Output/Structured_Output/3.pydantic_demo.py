from pydantic import BaseModel, EmailStr, Field, validate_email
from typing import Optional

class Student(BaseModel):

    name: str = 'nitish'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=3.78, description='A decimal value representing the cgpa of the student.')
new_student = {'age': '32', 'email': 'abc@gmail.com'}

student = Student(**new_student)

print(student)
print(student.name)
print(student.age)
print(student.email)
print(student.cgpa)

student_dict = dict(student)
print(student_dict['cgpa'])

student_json = student.model_dump_json()
print(student_json)
print(student_json[5])