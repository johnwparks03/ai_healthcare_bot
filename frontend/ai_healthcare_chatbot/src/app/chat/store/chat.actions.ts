import { createAction, props } from '@ngrx/store';
import { user_input, model_output } from '../models/chat';

export const PostUserInput = createAction(
  '[Chat] Post User Input',
  props<{ payload: user_input }>()
);
export const PostUserInputSuccess = createAction(
  '[Chat] Post User Input Success',
  props<{ response: model_output }>()
);
export const PostUserInputFailure = createAction(
  '[Chat] Post User Input Failure',
  props<{ error: any }>()
);
