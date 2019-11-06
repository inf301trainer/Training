import { Component, OnInit } from '@angular/core';
import { Http } from '@angular/http';

export const TASK_FILE = '/data/tasks/tasks.csv';

export const PROJECT = 0;
export const TITLE = 1;
export const TARGET_DATE = 2;
export const FINISH_DATE = 3;
export const STATUS = 4;
export const LEVEL = 5;
export const COMMENT = 6;

export class Task {
  private project: string;
  private title: string;
  private targetDate: string;
  private finishDate: string;
  private status: Number;
  private level: Number;
  private comment: string;

  children: Task[] = [];
  parent: Task = null;

  constructor(project, title, targetDate, finishDate, status, level, comment) {
    this.project = project;
    this.title = title;
    this.targetDate = targetDate;
    this.finishDate = finishDate;
    this.status = status;
    this.level = level;
    this.comment = comment;
  }

  getProject() {
    return this.project;
  }

  setProject(project: string) {
    this.project = project;
  }

  addChild(child: Task) {
    this.children.push(child);
  }

  setParent(parent: Task) {
    this.parent = parent;
  }
}

@Component({
  selector: 'app-task',
  templateUrl: './task.component.html',
  styleUrls: ['./task.component.css']
})

export class TaskComponent implements OnInit {
  title = 'Quản lí công việc';
  tasks: Task[] = [];
  
  constructor(private http: Http) { }

  ngOnInit() {
    this.http.get(TASK_FILE).subscribe((data: any) => {
      const lines = data._body.split('\n');

      let currentParentTask0 = null;
      let currentParentTask1 = null;

      for (let idx in lines) {
        const line = lines[idx];
        if (line) {
          const elements = line.split('\t');
          let currentTask = new Task(elements[PROJECT], elements[TITLE], elements[TARGET_DATE], elements[FINISH_DATE], elements[STATUS], elements[LEVEL], elements[COMMENT]);
          
          if (elements[LEVEL] == 0) {
            currentParentTask0 = currentTask;
            this.tasks.push(currentTask);
          }

          else if (elements[LEVEL] == 1) {
            currentParentTask1 = currentTask;
            currentTask.setProject(currentParentTask0.getProject());
            currentTask.setParent(currentParentTask0);
            currentParentTask0.addChild(currentTask);
          }

          else if (elements[LEVEL] == 2) {
            currentTask.setProject(currentParentTask0.getProject());
            currentTask.setParent(currentParentTask1);
            currentParentTask1.addChild(currentTask);
          }
        }
      }
    });
  }

}
