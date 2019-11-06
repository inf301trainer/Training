import { Component, OnInit } from '@angular/core';
import { LECTURES, LECTURE_LIST } from '../../lectures-list';
import { COURSES, COURSE_LIST } from '../../courses-list';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-md-lectures-list',
  templateUrl: './md-lectures-list.component.html',
  styleUrls: ['./md-lectures-list.component.css']
})
export class MdLecturesListComponent implements OnInit {
  lectureList = LECTURE_LIST;
  lectures = LECTURES;
  courses = COURSES;
  courseList = COURSE_LIST;
  course = {};

  constructor(private route: ActivatedRoute) { }

  ngOnInit() {
    const courseId = this.route.snapshot.paramMap.get('courseId');
    if (courseId) {
      this.course = COURSES[courseId];
      this.lectureList = [];
      LECTURE_LIST.forEach((item) => {
        if (this.lectures[item]['categories'].indexOf(courseId) >= 0) {
          this.lectureList.push(item);
        }
      });
    }
  }

}
