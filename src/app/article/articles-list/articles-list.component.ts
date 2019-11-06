import { Component, OnInit, Input } from '@angular/core';
import { ARTICLES, ARTICLE_VALUE_LIST, CATEGORIES, CATEGORY_LIST } from '../../articles-list';
import { PERIODS } from '../../periods-list';
import { ArticleComponent } from '../article.component';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-articles-list',
  templateUrl: './articles-list.component.html',
  styleUrls: ['./articles-list.component.css']
})
export class ArticlesListComponent implements OnInit {

  articles = ARTICLES;
  articleValueList = ARTICLE_VALUE_LIST;

  
  categories = CATEGORIES;
  categoryList = CATEGORY_LIST;

  periodKeys = Object.keys(PERIODS);
  
  // Display only some categories
  @Input() selectedCategories = ['all', 'essay', 'poem'];

  constructor(private route: ActivatedRoute) {
  }

  ngOnInit() {
    if (this.route.snapshot.data.categories) {
      this.selectedCategories = this.route.snapshot.data.categories;
    }
  }

  filterByPrivacy (article: ArticleComponent) {
    return !article.private;
  }

}
