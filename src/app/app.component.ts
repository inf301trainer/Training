import { Component } from '@angular/core';
import { ARTICLES } from './articles-list';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'BlogMigration';
  articles = ARTICLES;

  showLeftPanel = true;
}
