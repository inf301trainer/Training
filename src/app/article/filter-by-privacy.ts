import { Injectable, Pipe, PipeTransform } from '@angular/core';
import { ArticleComponent } from './article.component';

@Pipe({
  name: 'filterByPrivacy'
})

@Injectable()
export class FilterByPrivacyPipe implements PipeTransform {
  transform(items: ArticleComponent[], mode: string, field: string, value: string = null): any[] {
    if (!items) {
      return [];
    }

    if (mode === 'if') {
      return items.filter(it => it[field]);
    }

    if (mode === 'unless') {
      return items.filter(it => !it[field]);
    }

    if (mode === 'contains') {
      return items.filter(it => it[field].indexOf(value) >= 0);
    }

    if (mode === 'equal') {
      return items.filter(it => it[field] === value);
    }
  }
}
