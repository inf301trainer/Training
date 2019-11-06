import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpModule } from '@angular/http';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { RouterModule, Routes } from '@angular/router';
import { MarkdownModule, MarkedOptions, MarkedRenderer } from 'ngx-markdown';

import { NgbModule } from '@ng-bootstrap/ng-bootstrap';

import { AppComponent } from './app.component';
import { ArticleComponent } from './article/article.component';
import { HeaderComponent } from './header/header.component';
import { MenuComponent } from './menu/menu.component';
import { ArticlesListComponent } from './article/articles-list/articles-list.component';
import { MdLectureComponent } from './md-lecture/md-lecture.component';
import { MdLecturesListComponent } from './md-lecture/md-lectures-list/md-lectures-list.component';

import { FilterByPrivacyPipe } from './article/filter-by-privacy';
import { TaskComponent } from './task/task/task.component';

export function markedOptionsFactory(): MarkedOptions {
  const renderer = new MarkedRenderer();

  renderer.blockquote = (text: string) => {
    return '<blockquote class="blockquote"><p>' + text + '</p></blockquote>';
  };

  return {
    renderer: renderer,
    gfm: true,
    tables: true,
    breaks: true,
    pedantic: false,
    sanitize: false,
    smartLists: true,
    smartypants: false,
  };
}

const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'articles' },
  { path: 'articles', component: ArticlesListComponent },
  { path: 'article/:idx', component: ArticleComponent },
  { path: 'poems', component: ArticlesListComponent, data: {categories: ['poem']} },
  { path: 'essays', component: ArticlesListComponent, data: {categories: ['essay']} },
  { path: 'poems/thuat-hung', component: ArticlesListComponent, data: {categories: ['thuat-hung']} },
  { path: 'poems/phong-tac', component: ArticlesListComponent, data: {categories: ['phong-tac']} },
  { path: 'poems/kai', component: ArticlesListComponent, data: {categories: ['kai']} },
  // COURSES
  { path: 'courses', component: MdLecturesListComponent },
  { path: 'course/:courseId', component: MdLecturesListComponent },
  { path: 'lecture/:idx', component: MdLectureComponent },
  // TASKS
  { path: 'tasks', component: TaskComponent }
];
@NgModule({
  declarations: [
    AppComponent,
    ArticleComponent,
    HeaderComponent,
    MenuComponent,
    ArticlesListComponent,
    MdLectureComponent,
    MdLecturesListComponent,

    FilterByPrivacyPipe,

    TaskComponent
  ],
  imports: [
    BrowserModule,
    HttpModule,
    HttpClientModule,
    FormsModule,
    NgbModule,
    RouterModule.forRoot(routes, {enableTracing: true}),
    MarkdownModule.forRoot({ loader: HttpClient, markedOptions: {
      provide: MarkedOptions,
      useFactory: markedOptionsFactory,
    }, })
  ],
  providers: [],
  bootstrap: [AppComponent]
})

export class AppModule { }
