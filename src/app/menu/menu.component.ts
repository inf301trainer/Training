import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-menu',
  templateUrl: './menu.component.html',
  styleUrls: ['./menu.component.css']
})
export class MenuComponent implements OnInit {
  categories = [
    ['Về tôi', 'cv', 0],
    ['Các khoá học', 'courses', 0],
    ['DSC101 - Machine Learning cơ bản với Python', 'course/dsc101', 1],
    ['DSC102 - Machine Learning cơ bản [2]', 'course/dsc102', 1],
    ['DSC111 - Deep Learning cơ bản', 'course/dsc111', 1],
    ['Ứng dụng cá nhân', 'apps', 0],
    ['Quản lí công việc', 'tasks', 1],
    ['Theo dõi tài chính', 'finance', 1],
    ['Vốn từ ngoại ngữ', 'voca', 1],
    ['Tiếng Anh', 'e-voca', 2],
    ['Tiếng Pháp', 'f-voca', 2],
    ['Tiếng Trung Quốc', 'c-voca', 2],
    ['Risk (trò chơi) - Đại chiến thế giới', 'risk', 1],
    ['Góc viết lách', 'articles', 0],
    ['Văn xuôi', 'essays', 1],
    ['Thơ', 'poems', 1],
    ['Thuật hứng', 'poems/thuat-hung', 2],
    ['Phóng tác', 'poems/phong-tac', 2],
    ['Kai', 'poems/kai', 2]
  ];

  constructor() { }

  ngOnInit() {
  }

}
