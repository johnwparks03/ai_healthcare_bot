import { Injectable } from '@angular/core';
import { Actions, createEffect, ofType } from '@ngrx/effects';
import { ChatApiService } from '../services/chat-api.service';
import * as ChatActions from './chat.actions';
import { catchError, mergeMap, of, map } from 'rxjs';

@Injectable()
export class ChatEffects {
  postUserInput$ = createEffect(() =>
    this.actions$.pipe(
      ofType(ChatActions.PostUserInput),
      mergeMap(({ payload }) =>
        this.chatApiService.postUserInput(payload).pipe(
          map((response) => ChatActions.PostUserInputSuccess({ response })),
          catchError((error) => of(ChatActions.PostUserInputFailure({ error })))
        )
      )
    )
  );

  constructor(
    private actions$: Actions,
    private chatApiService: ChatApiService
  ) {}
}
