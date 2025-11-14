import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { model_output, user_input } from '../models/chat';

@Injectable({
  providedIn: 'root',
})
export class ChatApiService {
  constructor(private http: HttpClient) {}

  postUserInput(input: user_input) {
    return this.http.post<model_output>('http://localhost:8000/predict', input);
  }
}
