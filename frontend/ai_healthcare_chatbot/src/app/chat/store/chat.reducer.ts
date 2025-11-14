import { createReducer, on } from '@ngrx/store';
import * as ChatActions from './chat.actions';
import { user_input, model_output } from '../models/chat';

export interface State {
    loading: boolean;
  model_outputs: model_output[];
}

export const initialState: State = {
    loading: false,
  model_outputs: [],
};

export const chatReducer = createReducer(
    initialState,
    on(ChatActions.PostUserInput, (state) => ({
        ...state,
        loading: true
    })),
    on(ChatActions.PostUserInputSuccess, (state, { response }) => ({
        ...state,
        loading: false,
        model_outputs: [...state.model_outputs, response]
    })),
    on(ChatActions.PostUserInputFailure, (state, { error }) => ({
        ...state,
        loading: false
    }))
);
