import { Component, OnInit, Input } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { LECTURES } from '../lectures-list';

@Component({
  selector: 'app-md-lecture',
  templateUrl: './md-lecture.component.html',
  styleUrls: ['./md-lecture.component.css']
})
export class MdLectureComponent implements OnInit {
  @Input() title: string;
  @Input() contentTemplate: string;
  @Input() date: string;
  @Input() categories: string[];
  @Input() private = false;
  content = '';

  constructor(private route: ActivatedRoute) {
  }

  ngOnInit() {
    const idx = this.route.snapshot.paramMap.get('idx');
    if (idx) {
      const lecture = LECTURES[idx];
      this.title = lecture.title;
      this.contentTemplate = lecture.contentTemplate;
      this.date = lecture.date;
      this.categories = lecture.categories;
      this.private = lecture.private;
    }

    if (!this.contentTemplate) {
      return;
    }
  }

  reformatDate() {
    return this.date.slice(8, 10) + '-' + this.date.slice(5, 7) + this.date.slice(0, 4);
  }
}
