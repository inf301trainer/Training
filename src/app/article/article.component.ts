import { Component, OnInit, Input } from '@angular/core';
import { Http } from '@angular/http';
import { ActivatedRoute } from '@angular/router';
import { ARTICLES } from '../articles-list';

@Component({
  selector: 'app-article',
  templateUrl: './article.component.html',
  styleUrls: ['./article.component.css']
})

export class ArticleComponent implements OnInit {
  @Input() title: string;
  @Input() contentTemplate: string;
  @Input() date: string;
  @Input() categories: string[];
  @Input() private = false;
  content = '';

  constructor(private http: Http, private route: ActivatedRoute) {
  }

  ngOnInit() {
    const idx = this.route.snapshot.paramMap.get('idx');
    const article = ARTICLES[idx];
    this.title = article.title;
    this.contentTemplate = article.contentTemplate;
    this.date = article.date;
    this.categories = article.categories;
    this.private = article.private;

    if (!this.contentTemplate) {
      return;
    }

    this.http.get(this.contentTemplate).subscribe((html: any) => {
      this.content = html._body;
    });
  }

}
