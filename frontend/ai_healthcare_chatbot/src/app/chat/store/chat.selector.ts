import { createFeatureSelector, createSelector } from "@ngrx/store";
import { State } from "./chat.reducer";

export const selectChatState = createFeatureSelector<State>('chat');

export const selectAllModelOutputs = createSelector(
    selectChatState,
    (state) => state.model_outputs
)