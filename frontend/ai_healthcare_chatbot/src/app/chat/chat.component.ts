import {
    AfterViewChecked,
    AfterViewInit,
    Component,
    ElementRef,
    OnDestroy,
    ViewChild,
} from '@angular/core';
import { selectAllModelOutputs } from './store/chat.selector';
import { Store } from '@ngrx/store';
import { user_input } from './models/chat';
import * as ChatActions from './store/chat.actions';

@Component({
    selector: 'chat_component',
    templateUrl: './chat.component.html',
    styleUrls: ['./chat.component.css'],
})
export class ChatComponent implements AfterViewChecked {
    inputValue = '';
    outputs$ = this.store.select(selectAllModelOutputs);
    @ViewChild('responsesContainer') container!: ElementRef;

    private shouldScroll = false;

    constructor(private store: Store) {
        this.outputs$.subscribe(() => {
            this.shouldScroll = true;
        });
    }

    onSend() {
        const payload: user_input = { user_input: this.inputValue };
        this.inputValue = '';
        this.store.dispatch(ChatActions.PostUserInput({ payload }));
    }

    ngAfterViewChecked() {
        if (this.shouldScroll && this.container) {
            const el = this.container.nativeElement;
            el.scrollTop = el.scrollHeight;
            this.shouldScroll = false;
        }
    }
}
